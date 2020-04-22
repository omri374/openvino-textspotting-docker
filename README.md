## text-spotting-docker: Docker installation of OpenVino's text-spotting model

This repo contains a wrapper around the OpenVino text spotting model. 
It packages all the requirements in a docker container and exposes a REST API for calling the model for inference.

### Development

### Install dependencies
```shell script
# Install Dependancies:
sudo apt-get update && sudo apt-get install -yy wget libjpeg-turbo8-dev libz-dev python3-pip python3-venv git-lfs
# Create a virtual environment if doesn't exist
python3 -venv ~/envs/text_spotting/
# Activate virtual environment
source  ~/envs/text_spotting/bin/activate
# Install in dev mode
pip install -e .
# Run tests
pytest

## Setting up OpenVino (intel inference engine)
sudo scripts/install-openvino.sh
scripts/install-openvino-python.sh
```

#### Install package locally
```shell script
pip install -e .
```



#### Download models
```shell script
text_spotting_get_models
```

#### Run server
```shell script
text_spotting
```

#### Test
```shell script
pytest
```


##### Request object
/run_ocr:
```
{
    "image": "base64 jpeg encoded image"
}
```

##### Response
```
{[
    "text": str,  # identified text
    "coords": {"left": float, "top": float, "right": float, "bottom": float],  # Coordinates of bounding box of text
    "score": float  # Confidence value
]}
```
---

### Build Docker

```sh
sudo docker build -t temp/text_spotting_ocr .
```

### Run Docker

```
sudo docker run temp/text_spotting_ocr
```
References:

1. Full dockerfile in rafael's repo: https://github.com/giladfr-rnd/monitors-cv/blob/master/full.Dockerfile
1. OpenVino model server in Docker: https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.mdThis contains code for deployment an OpenVino model to K8S (for example) using grpc
1. How to install OpenVino on Docker: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html

Run the original OpenVino demo:
```
python text_spotting/text_spotting_demo.py -m_m models/FP32/text-spotting-0002-detector.xml -m_te models/FP32/text-spotting-0002-recognizer-encoder.xml -m_td models/FP32/text-spotting-0002-recognizer-decoder.xml -i data/photo.jpg --no_show --raw_output_message --no_track
```
