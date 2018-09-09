#FROM gcr.io/tpu-test-001/gpu-base-image
FROM tensorflow/tensorflow:latest-gpu

COPY deepxor /tpu-test-001
WORKDIR /tpu-test-001

