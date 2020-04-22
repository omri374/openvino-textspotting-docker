import logging
import os
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


class ModelHandler:
    """
    Downloads and stores OCR models
    """

    @staticmethod
    def get_models(models_path="/text_spotting_model/models.yaml"):
        models_list = yaml.safe_load(open(os.path.dirname(__file__) + models_path, 'r'))['files']
        print(f"Loaded model list: {models_list}")
        models = {}
        for model in models_list:
            models[model['name']] = ModelHandler.download_model(model['source'], model['name'])
        return models

    @staticmethod
    def download_model(url, filename, basedir="../models"):
        to_path = str(Path(os.path.dirname(__file__), basedir, filename).resolve())

        if not os.path.exists(to_path):
            ModelHandler.__download(url=url, to_path=to_path, file_name=filename)

        logging.info("model found.")
        return to_path

    @staticmethod
    def __download(url, to_path, file_name):
        print(f"Downloading model file {file_name}")
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
        print(f"Finished downloading model file {file_name}")


def download_models():
    ModelHandler.get_models()
    return 0
