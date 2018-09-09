#FROM gcr.io/tpu-test-001/gpu-base-image
FROM gcr.io/tensorflow/tensorflow-gpu

COPY deepxor /tpu-test-001
WORKDIR /tpu-test-001

