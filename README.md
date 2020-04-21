## text-spotting-docker: Docker installation of OpenVino's text-spotting model

This repo contains a wrapper around the OpenVino text spotting model. 
It packages all the requirements in a docker container and exposes a REST API for calling the model for inference.

### Development

#### Install locally
```shell script
pip install -e .
```

#### Download models
````shell script
text_spotting_get_models
````

#### Run server
````shell script
text_spotting
````

References:

1. Full dockerfile in rafael's repo: https://github.com/giladfr-rnd/monitors-cv/blob/master/full.Dockerfile
1. OpenVino model server in Docker: https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.mdThis contains code for deployment an OpenVino model to K8S (for example) using grpc
1. How to install OpenVino on Docker: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html

Run the original OpenVino demo:
```
python text_spotting/text_spotting_demo.py -m_m models/FP32/text-spotting-0002-detector.xml -m_te models/FP32/text-spotting-0002-recognizer-encoder.xml -m_td models/FP32/text-spotting-0002-recognizer-decoder.xml -i data/photo.jpg --no_show --raw_output_message --no_track
```

Still missing:

1. Dockerize everything
1. Define agreed API
   1. Change image read to input
   1. Define output structure
