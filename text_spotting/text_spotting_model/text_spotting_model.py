import logging as log
import os
import sys
import time

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

from text_spotting import ModelHandler
from .tracker import StaticIOUTracker

SOS_INDEX = 0
EOS_INDEX = 1


class TextSpottingModel:

    def __init__(self, device='CPU', track=False, visualize=False, prob_threshold=0.3, max_seq_len=10,
                 iou_threshold=0.4, model_type='FP32', rgb2bgr=True, performance_counts=False, verbose=True):

        assert (model_type == 'FP32') or (model_type == 'FP16')

        mask_rcnn_model_xml = ModelHandler.get_models()['{}/text-spotting-0002-detector.xml'.format(model_type)]
        mask_rcnn_model_bin = ModelHandler.get_models()['{}/text-spotting-0002-detector.bin'.format(model_type)]

        text_enc_model_xml = ModelHandler.get_models()['{}/text-spotting-0002-recognizer-encoder.xml'.format(model_type)]
        text_enc_model_bin = ModelHandler.get_models()['{}/text-spotting-0002-recognizer-encoder.bin'.format(model_type)]

        text_dec_model_xml = ModelHandler.get_models()['{}/text-spotting-0002-recognizer-decoder.xml'.format(model_type)]
        text_dec_model_bin = ModelHandler.get_models()['{}/text-spotting-0002-recognizer-decoder.bin'.format(model_type)]

        # Plugin initialization for specified device and load extensions library if specified.
        log.info('Creating Inference Engine...')
        ie = IECore()
        # Read IR
        log.info('Loading network files:\n\t{}\n\t{}'.format(mask_rcnn_model_xml, mask_rcnn_model_bin))
        mask_rcnn_net = IENetwork(model=mask_rcnn_model_xml, weights=mask_rcnn_model_bin)

        log.info('Loading network files:\n\t{}\n\t{}'.format(text_enc_model_xml, text_enc_model_bin))
        text_enc_net = IENetwork(model=text_enc_model_xml, weights=text_enc_model_bin)

        log.info('Loading network files:\n\t{}\n\t{}'.format(text_dec_model_xml, text_dec_model_bin))
        text_dec_net = IENetwork(model=text_dec_model_xml, weights=text_dec_model_bin)

        supported_layers = ie.query_network(mask_rcnn_net, 'CPU')
        not_supported_layers = [layer for layer in mask_rcnn_net.layers.keys() if layer not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error('Following layers are not supported by the plugin for specified device {}:\n {}'.
                      format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(mask_rcnn_net.inputs.keys()), \
            'Demo supports only topologies with the following input keys: {}'.format(', '.join(required_input_keys))
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks', 'text_features'}
        assert required_output_keys.issubset(mask_rcnn_net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        n, c, h, w = mask_rcnn_net.inputs['im_data'].shape
        assert n == 1, 'Only batch 1 is supported by the demo application'
        self.shape = [n, c, h, w]
        self.verbose = verbose
        log.info('Loading IR to the plugin...')
        self.mask_rcnn_exec_net = ie.load_network(network=mask_rcnn_net, device_name=device, num_requests=2)
        self.text_enc_exec_net = ie.load_network(network=text_enc_net, device_name=device)
        self.text_dec_exec_net = ie.load_network(network=text_dec_net, device_name=device)
        self.hidden_shape = text_dec_net.inputs['prev_hidden'].shape
        self.prob_threshold = prob_threshold
        self.tracker = None
        self.visualizer = None
        self.iou_threshold = iou_threshold
        self.max_seq_len = max_seq_len
        self.rgb2bgr = rgb2bgr
        self.perf_counts = performance_counts
        # self.device_names = get_fields_info()  # Change if we want to explicit set device parameters (respirator, ivac, monitor)
        self.device_names = None
        if track:
            self.tracker = StaticIOUTracker()
        log.info('Model ready...')

    def predict(self, frame):
        """
        returns: texts, boxes, scores, frame
        boxes are [left, top, right, bottom]
        """
        [n, c, h, w] = self.shape
        # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
        scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
        input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        if self.rgb2bgr:
            input_image = input_image[:, :, ::-1]  # reverse channels order

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)
        del n, c, h, w
        # Run the net.
        inf_start = time.time()
        log.info('running main network')
        outputs = self.mask_rcnn_exec_net.infer({'im_data': input_image, 'im_info': input_image_info})
        log.info('main network finished')
        if len(outputs['boxes']) == 0:
            return [], [], [], []
        # Parse detection results of the current request
        boxes = outputs['boxes']
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        raw_masks = outputs['raw_masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > self.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        # initialize text_features_secondary
        text_features_secondary = [None for t in text_features]
        matches_secondary = [None for t in text_features]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y

        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            raw_cls_mask = raw_mask[cls, ...]
            mask = self.segm_postprocess(box, raw_cls_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        texts = []
        alphabet = '  0123456789abcdefghijklmnopqrstuvwxyz'
        for feature in text_features:
            feature = self.text_enc_exec_net.infer({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for i in range(self.max_seq_len):
                decoder_output = self.text_dec_exec_net.infer({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature})
                symbols_distr = decoder_output['output']
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += alphabet[prev_symbol_index]
                hidden = decoder_output['hidden']

            texts.append(text)
            print(f"Found texts: {text}")

        inf_end = time.time()
        inf_time = inf_end - inf_start

        render_start = time.time()

        if len(boxes) and self.verbose:
            log.info('Detected boxes:')
            log.info('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
            for box, cls, score, mask in zip(boxes, classes, scores, masks):
                log.info('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

        # Get instance track IDs.
        masks_tracks_ids = None
        if self.tracker is not None:
            masks_tracks_ids = self.tracker(masks, classes)

        render_time = 0
        # Visualize masks.
        # frame = visualizer(frame, boxes, classes, scores, masks, texts, masks_tracks_ids)

        # Draw performance stats.
        inf_time_message = 'Inference and post-processing time: {:.3f} ms'.format(inf_time * 1000)
        render_time_message = 'OpenCV rendering time: {:.3f} ms'.format(render_time * 1000)

        # Print performance counters.
        if self.perf_counts:
            perf_counts = self.mask_rcnn_exec_net.requests[0].get_perf_counts()
            log.info('Performance counters:')
            print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format('name', 'layer_type', 'exet_type', 'status',
                                                              'real_time, us'))
            for layer, stats in perf_counts.items():
                log.debug('{:<70} {:<15} {:<15} {:<15} {:<10}'.format(layer, stats['layer_type'], stats['exec_type'],
                                                                      stats['status'], stats['real_time']))

        render_end = time.time()
        render_time = render_end - render_start
        log.info(f"Render time = {render_time}")

        return texts, boxes, scores, frame

    @staticmethod
    def segm_postprocess(box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
        extended_box = TextSpottingModel.expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                (x0 - extended_box[0]):(x1 - extended_box[0])]
        return im_mask

    @staticmethod
    def expand_box(box, scale):
        w_half = (box[2] - box[0]) * .5
        h_half = (box[3] - box[1]) * .5
        x_c = (box[2] + box[0]) * .5
        y_c = (box[3] + box[1]) * .5
        w_half *= scale
        h_half *= scale
        box_exp = np.zeros(box.shape)
        box_exp[0] = x_c - w_half
        box_exp[2] = x_c + w_half
        box_exp[1] = y_c - h_half
        box_exp[3] = y_c + h_half
        return box_exp