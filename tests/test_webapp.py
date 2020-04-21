import base64
import os
from pathlib import Path

from flask import url_for


def test_ping(client):
    res = client.get(url_for("ping"))
    assert res.data == b"pong"


def test_ocr(client):
    src_image = open(Path(os.path.dirname(__file__), "../data/out1.png").resolve(), "rb").read()
    assert len(src_image) > 0

    image_buffer = base64.encodebytes(src_image).decode()
    data = {"image": image_buffer}

    res = client.post(url_for("run_ocr"), json=data)
    assert res.status_code == 200
