import os
import logging
import base64
import json
import copy
import random

from flask import Flask, request, jsonify
import numpy as np
import imageio

from .text_spotting_model import TextSpottingModel


class Server:

    def __init__(self, log_level=logging.DEBUG):
        self.app = Flask(__name__)
        self.app.logger.setLevel(log_level)
        self.text_spotting = TextSpottingModel()
        self.threshold = float(os.environ.get("OCR_THRESHOLD", "0.8"))

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
            data =request.json
            image = None
            if "image" in data:
                image_data = base64.decodebytes(data["image"].encode())
                image = np.asarray(imageio.imread(image_data))
            else:
                return jsonify("image not found in request object")

            logging.debug(f'id: {data.get("monitorId")}:{data.get("imageId")}')
            segments = copy.deepcopy(data.get("segments", []))

            # Call model
            try:
                texts, boxes, scores, _ = self.text_spotting.forward(image)

                # Parse output
                should_log = False
                if texts:
                    for eb, text, score in zip(expected_boxes, texts, scores):
                        if score > self.threshold:
                            if segments[eb["index"]].get("value") != text:
                                should_log = True
                            segments[eb["index"]]["value"] = text
                            segments[eb["index"]]["score"] = float(score)
                            segments[eb["index"]]["source"] = "server"

                if should_log or random.randint(0, 100) == 0 or not texts:
                    self.resultsLogger.log_ocr(image_data, data["segments"],
                                               {'expected': expected_boxes, 'texts': texts,
                                                'scores': [float(s) for s in scores]})
                for s in segments:
                    if 'value' not in s:
                        s['value'] = None
                logging.debug(f"Detections: {segments}")
                return json.dumps(segments), 200, {"content-type": "application/json"}
            except Exception as e:
                return jsonify(f"Error running OCR. Exception: {e}")
