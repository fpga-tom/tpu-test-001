FROM ubuntu:bionic

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        sudo \
        gnupg \
        lsb-release \
        ca-certificates \
        build-essential \
        git \
        python3 \
        python3-pip \
        python3-setuptools && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    pip3 install tensorflow==1.9 && \
    pip3 install google-cloud-storage

COPY mnist /mnist
WORKDIR /mnist
