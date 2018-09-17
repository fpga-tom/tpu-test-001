# On ps0.example.com:
#WORKER_HOSTS=localhost:2223,localhost:2224,localhost:2225,localhost:2226,localhost:2227,localhost:2228,localhost:2229,localhost:2230
WORKERS=`seq 0 7`
WORKER_HOSTS=''
for idx in $WORKERS; do
	if [ "$idx" -eq "0" ]; then
		WORKER_HOSTS=localhost:$((2223+$idx))
	else
		WORKER_HOSTS=$WORKER_HOSTS,localhost:$((2223+$idx))
	fi
done

MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-async-gd-6
CUDA_VISIBLE_DEVICES='' python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=$WORKER_HOSTS \
     --job_name=ps --task_index=0 --model_dir=$MODEL_DIR --train_steps=1000000 --learning_rate=0.01 &
echo "kill $!" > stop.sh
# On worker0.example.com:
for idx in $WORKERS; do
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=$WORKER_HOSTS \
     --job_name=worker --task_index=$idx \
     --rolls=64 --rolls_len=32 --model_dir=$MODEL_DIR --train_steps=1000000 --learning_rate=0.01 &
echo "kill $!" >> stop.sh
done
chmod u+x stop.sh
