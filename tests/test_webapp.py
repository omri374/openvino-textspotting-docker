from pathlib import Path
import json

from flask import url_for
import imageio
import cv2
import numpy as np
import os
import base64
import pytest



def test_ping(client):

    res = client.get(url_for("ping"))
    assert res.data == b"pong"

def test_ocr(client):
    src_image = open(Path(os.path.dirname(__file__),"../data/out1.png").resolve(), "rb").read()
    assert len(src_image) > 0

    image_buffer = base64.encodebytes(src_image).decode()
    data = {"image": image_buffer}

    res = client.post(url_for("run_ocr"), json=data)
    #if res.status_code == 200:
    res_image = np.asarray(imageio.imread(res.data))
    up_image = imageio.imread(Path(os.path.dirname(__file__),"../data/out1.png").resolve())
    assert np.median(np.abs(res_image - up_image)) < 2.0

    #else:
    #    raise Exception(f"Failed to call ocr service. Response = {res}")
