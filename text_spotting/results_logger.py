import json
import os
import time
from pathlib import Path


class ResultsLogger:

    def __init__(self):
        self.basedir = '../../logs'
        self.index = 0

    def log_ocr(self, image, texts, boxes, scores, image_id=None):
        self.index += 1
        folder_name = Path(self.basedir, time.strftime("%Y_%m_%d_%H")).resolve()
        os.makedirs(folder_name, exist_ok=True)
        file_name = Path(folder_name, f"{self.index:09}").resolve()
        if image_id:
            file_name = Path(folder_name, image_id).resolve()

        json.dump({'server_ocr': {"texts": texts, "boxes": boxes.tolist(), "scores": scores.tolist()}},
                  open(f'{file_name}.json', 'w'))

        if image is not None:
            with open(str(file_name) + '.jpg', 'wb') as f:
                f.write(image)
