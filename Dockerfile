FROM ubuntu:18.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -yy wget libjpeg-turbo8-dev libz-dev python3-pip python3-venv git python3-tk &&  rm -rf /var/lib/apt/lists/*
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
COPY .git /opt/app/.git
RUN pip install -e .[full] --no-cache-dir
RUN pytest
RUN text_spotting_get_models
CMD text_spotting
