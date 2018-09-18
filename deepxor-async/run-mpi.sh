MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-async-gd-10
mpirun 	-np 2 \
	-bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-mca pml ob1 -mca btl ^openib \
	python3 cube.py  --rolls=64 --rolls_len=32 --model_dir=$MODEL_DIR --train_steps=1000000 --learning_rate=0.01 --train_steps_per_eval=100 --batch_size=32
