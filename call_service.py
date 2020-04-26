import base64
import json
import os
import sys
import time
from pathlib import Path

import click
import requests


@click.command()
@click.option("--file_path", default="data/out1.png", help="Location of image to run model on")
@click.option("--service_url", default="http://128.0.0.1:8081/run_ocr", help="Url and port of Text Spotting OCR service")
def call_ocr(file_path, service_url):
    src_image = open(Path(os.path.dirname(__file__), file_path).resolve(), "rb").read()
    image_buffer = base64.encodebytes(src_image).decode()
    data = {"image": image_buffer}
    click.echo(f"Calling Text Spotting OCR at {service_url} with image from {src_image}")
    start_time = time.time()
    try:
        response = requests.post(url=service_url, json=data)
        print(f"Status code = {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print("Failed to call model. Http Error:", errh)
        sys.exit(1)
    except requests.exceptions.ConnectionError as errc:
        print("Failed to call model. Error Connecting:", errc)
        sys.exit(1)
    except requests.exceptions.Timeout as errt:
        print("Failed to call model. Timeout Error:", errt)
        sys.exit(1)
    except requests.exceptions.RequestException as err:
        print("Failed to call model. Error retrieving model", err)
        sys.exit(1)
    duration = time.time() - start_time
    print(f"Request duration: {duration}")
    response_obj = json.loads(response.text)
    print(f"Response: {response_obj}")
    return response_obj


if __name__ == "__main__":
    call_ocr()
