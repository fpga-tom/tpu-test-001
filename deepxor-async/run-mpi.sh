MODEL_DIR=/home/tomas/Documents/deepxor-async-gd-9
/usr/lib64/openmpi/bin/mpirun -np 8 python3 cube.py  --rolls=128 --rolls_len=64 --model_dir=$MODEL_DIR --train_steps=1000000 --learning_rate=0.01 --train_steps_per_eval=200
