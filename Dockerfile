FROM gcr.io/tensorflow/tpu-models:r1.9

COPY mnist /mnist
WORKDIR /mnist
