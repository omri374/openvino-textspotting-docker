[![Build Status](https://tfsprodweu3.visualstudio.com/Ac38be8bc-3e13-4419-bad2-46535e77c903/Rafael/_apis/build/status/Continuous%20Integration?branchName=master)](https://tfsprodweu3.visualstudio.com/Ac38be8bc-3e13-4419-bad2-46535e77c903/Rafael/_build/latest?definitionId=183&branchName=master)

## text-spotting-docker: Docker installation of OpenVino's text-spotting model

This repo contains a wrapper around the OpenVino text spotting model. 
It packages all the requirements in a docker container and exposes a REST API for calling the model for inference.

### Development

### Install dependencies (Ubuntu)
```shell script
# Install Dependencies:
sudo apt-get update && sudo apt-get install -yy wget python3-pip python3-venv
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
```
text_spotting_get_models
```

#### Run server
```
text_spotting
```


#### Call service

```
get_ocr --file_path data/out1.png --service_url http://128.0.0.1:8081/run_ocr
```


#### Test
```
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
    "coords": {"left": float, "top": float, "right": float, "bottom": float},  # Coordinates of bounding box of text
    "score": float  # Confidence value
]}
```
---

### Build Docker

```sh
sudo docker build -t omri374/text_spotting_ocr .
```

### Run Docker (interactively)

```
sudo docker run -it -p 8080:8088 omri374/text_spotting_ocr
```

#### Test docker by calling the web service
`call_service` accepts two parameters: 
- file_path: Location of image to run model on
- service_url: Web service URL

Example call:

```
python call_service.py --service_url http://0.0.0.0:8080/run_ocr --file_path data/out1.png
```

Run the original OpenVino demo:
```
python text_spotting/text_spotting_demo.py -m_m models/FP32/text-spotting-0002-detector.xml -m_te models/FP32/text-spotting-0002-recognizer-encoder.xml -m_td models/FP32/text-spotting-0002-recognizer-decoder.xml -i data/photo.jpg --no_show --raw_output_message --no_track
```


### References:

1. Many parts in this repo are based on [this repo](https://github.com/giladfr-rnd/monitors-cv/)
1. [OpenVino model server in Docker](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md). This contains code for deployment an OpenVino model to K8S (for example) using grpc
1. [How to install OpenVino on Docker](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html)
