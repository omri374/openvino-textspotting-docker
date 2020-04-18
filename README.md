# text-spotting-docker
Docker installation of OpenVino's text-spotting model

References:

1. Full dockerfile in rafael's repo: https://github.com/giladfr-rnd/monitors-cv/blob/master/full.Dockerfile
1. OpenVino model server in Docker: https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.mdThis contains code for deployment an OpenVino model to K8S (for example) using grpc
1. How to install OpenVino on Docker: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html

Current Dockerfile:
```sh
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -yy wget libzbar0 libjpeg-turbo8-dev libz-dev python3-pip python3-venv git python3-tk &&  rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -mvenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt/app/
RUN pip install -U wheel setuptools_scm setuptools
ADD requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
ADD scripts/ /opt/app/scripts/
RUN scripts/install-openvino.sh
RUN scripts/install-openvino-python.sh
COPY . /opt/app
RUN pip install -e .[full] --no-cache-dir
```

Download models:

### Text spotting:

```
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-detector/FP32/text-spotting-0002-detector.xml
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-detector/FP32/text-spotting-0002-detector.bin
https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-recognizer-decoder/FP32/text-spotting-0002-recognizer-decoder.xml
https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-recognizer-decoder/FP32/text-spotting-0002-recognizer-decoder.bin
https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-recognizer-encoder/FP32/text-spotting-0002-recognizer-encoder.bin
https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-spotting-0002-recognizer-encoder/FP32/text-spotting-0002-recognizer-encoder.xml

```

### Text detection

```
https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/text-detection-0003/FP32/
```


Run model:
```
python text_spotting_demo.py -m_m ../models/text-spotting-0002-detector.xml -m_te ../models/text-spotting-0002-recognizer-encoder.xml -m_td ../models/text-spotting-0002-recognizer-decoder.xml -i photo.png
```
