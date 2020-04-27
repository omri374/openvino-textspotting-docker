import base64
import json
import logging
import os
import random

import imageio
import numpy as np
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

from text_spotting import ResultsLogger, ModelHandler
from .text_spotting_model import TextSpottingModel


class Server:

    def __init__(self, log_level=logging.DEBUG, log_to_file=False):
        """
        A Flask based server which serves the IntelVino Text Spotting model. Currently exposes two services:
        ping (GET) and run_ocr(POST)
        :param log_level: Level for logging. Default is DEBUG
        :param log_to_file: Whether a sample of requests and responses should be logged to file. Default is False
        """
        self.app = Flask(__name__)
        self.app.logger.setLevel(log_level)
        self.text_spotting = TextSpottingModel()
        self.results_logger = ResultsLogger()
        self.threshold = float(os.environ.get("OCR_THRESHOLD", "0.8"))
        self.log_to_file = log_to_file

        @self.app.route('/ping/')
        def ping() -> str:
            """
            ping
            ---
            description:  get a pong
            """
            return 'pong'

        @self.app.route('/run_ocr', methods=["POST"])
        def run_ocr():
            """
            run_ocr
            ---
            description: Accepts a request object:
            {
                "image": "base64 jpeg encoded image"
            }
            Returns a json with identified text readings, including bounding boxes per text and confidence (score):
            {[
                "text": str,  # identified text
                "coords": {"left": float, "top": float, "right": float, "bottom": float},
                "score": float  # Confidence value
            ]}

            """
            data = request.json
            image = None
            if "image" in data:
                image_data = base64.decodebytes(data["image"].encode())
                image = np.asarray(imageio.imread(image_data))
            else:
                return jsonify("image not found in request object")

            logging.debug(f'id: {data.get("monitorId")}:{data.get("imageId")}')

            try:
                # Call model
                texts, boxes, scores, _ = self.text_spotting.predict(image)

                if self.log_to_file:
                    # Log results + image to file
                    self.log_result(boxes, image_data, scores, texts)

                # Create response object
                results = self.create_response(boxes, scores, texts)
                return json.dumps(results), 200, {"content-type": "application/json"}

            except Exception as e:
                logging.error(f"Fatal error on calling the model", exc_info=True)
                return jsonify(f"Error running OCR. Exception: {e}"), 500, {"content-type": "application/json"}

    def create_response(self, boxes, scores, texts):
        results = []
        for text, box, score in zip(texts, boxes, scores):
            if score < self.threshold:
                continue

            coords = {
                'left': float(box[0]),
                'top': float(box[1]),
                'right': float(box[2]),
                'bottom': float(box[3])
            }
            results.append({'text': text,
                            'coords': coords,
                            'score': float(score)})
        return results

    def log_result(self, boxes, image_data, scores, texts):
        should_log = False
        if texts:
            for eb, text, score in zip(boxes, texts, scores):
                if score > self.threshold:
                    should_log = True
        if should_log or random.randint(0, 100) == 0 or not texts:
            self.results_logger.log_ocr(image_data, texts, boxes, scores)


def init_logs():
    log_level_name = os.environ.get('CVMONITOR_LOG_LEVEL', 'DEBUG')
    log_level = logging.DEBUG
    if log_level_name == 'INFO':
        log_level = logging.INFO
    if log_level_name == 'WARNING':
        log_level = logging.WARNING
    if log_level_name == 'ERROR':
        log_level = logging.ERROR

    for logger in (logging.getLogger(),):
        logger.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return log_level


def main():
    #log_level = init_logs()
    host = os.environ.get('TEXT_SPOTTING_HOST', '0.0.0.0')
    port = int(os.environ.get('TEXT_SPOTTING_PORT', '8088'))
    log_to_file = bool(os.environ.get("LOG_TO_FILE", "False"))
    server = Server(logging.INFO, log_to_file)
    logging.info('checking if model exists locally:')
    ModelHandler.get_models()
    logging.info(f'serving on http://{host}:{port}/')
    WSGIServer((host, port), server.app).serve_forever()
