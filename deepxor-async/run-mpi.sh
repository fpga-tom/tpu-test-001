MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-async-gd-9
mpirun -np 8 python3 cube.py  --rolls=128 --rolls_len=64 --model_dir=$MODEL_DIR --train_steps=1000000 --learning_rate=0.01 --train_steps_per_eval=200
