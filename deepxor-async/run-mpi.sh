MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-allreduce-momentum-8
HOROVOD_FUSION_THRESHOLD=134217728 mpirun -x HOROVOD_FUSION_THRESHOLD \
	-np 1 \
	-bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-mca pml ob1 -mca btl ^openib \
	python3 cube.py  --rolls=64 --rolls_len=64 --model_dir=$MODEL_DIR --train_steps=2000000 --learning_rate=1e-2 --train_steps_per_eval=5 --batch_size=16 --checkpoint_steps=10000 --l2=1e-3
