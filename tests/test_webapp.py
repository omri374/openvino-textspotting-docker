import base64
import json
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


def test_response_struct_ok(client):
    src_image = open(Path(os.path.dirname(__file__), "../data/out1.png").resolve(), "rb").read()
    image_buffer = base64.encodebytes(src_image).decode()
    data = {"image": image_buffer}

    response = client.post(url_for("run_ocr"), json=data)

    for finding in json.loads(response.data):
        assert 'text' in finding
        assert 'coords' in finding
        assert 'score' in finding
        assert 'left' in finding['coords']
        assert 'right' in finding['coords']
        assert 'top' in finding['coords']
        assert 'bottom' in finding['coords']

        assert finding['coords']['left'] < finding['coords']['right']
        assert finding['coords']['top'] < finding['coords']['bottom']
