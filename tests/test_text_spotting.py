import base64
import os
from pathlib import Path

import imageio
import numpy as np

from text_spotting.text_spotting_model import TextSpottingModel


def test_model_correct_text_results():
    src_image = open(Path(os.path.dirname(__file__), "../data/out1.png").resolve(), "rb").read()
    model = TextSpottingModel()
    image = np.asarray(imageio.imread(src_image))

    texts, boxes, scores, _ = model.predict(image)
    assert len([text for text in texts if '120' == text]) == 1




