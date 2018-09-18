MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-allreduce-adam-2
HOROVOD_FUSION_THRESHOLD=134217728 mpirun -x HOROVOD_FUSION_THRESHOLD \
	-np 1 \
	-bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-mca pml ob1 -mca btl ^openib \
	python3 cube.py  --rolls=128 --rolls_len=64 --model_dir=$MODEL_DIR --train_steps=2000000 --learning_rate=0.001 --train_steps_per_eval=3 --batch_size=8 --checkpoint_steps=10000
