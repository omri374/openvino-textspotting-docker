import os
from pathlib import Path
import logging

from tqdm import tqdm
import requests


class ModelHandler:
    """
    Downloads and stores OCR models
    """

    @staticmethod
    def download_model(url, filename, basedir="../models"):
        to_path = str(Path(os.path.dirname(__file__), basedir, filename).resolve())

        if not os.path.exists(to_path):
            ModelHandler.__download(url=url, to_path=to_path, file_name=filename)

        logging.info("model found.")
        return to_path

    @staticmethod
    def __download(url, to_path, file_name):
        logging.info(f"Downloading model file {file_name}")
        os.makedirs(os.path.dirname(to_path), exist_ok=True)
        with open(to_path, "wb") as f:
            try:
                result = requests.get(url, stream=True)
                result.raise_for_status()
            except requests.exceptions.HTTPError as errh:
                print("Failed to download model. Http Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Failed to download model. Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Failed to download model. Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("Failed to download model. Error retrieving model", err)

            # Total size in bytes.
            total_size = int(result.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 Kilobyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=file_name)
            for data in result.iter_content(block_size):
                t.update(len(data))
                f.write(data)
            t.close()
        logging.info(f"Finished downloading model file {file_name}")
