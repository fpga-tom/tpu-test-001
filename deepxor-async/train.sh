# On ps0.example.com:
WORKER_HOSTS=localhost:2223,localhost:2224,localhost:2225,localhost:2226,localhost:2227,localhost:2228,localhost:2229,localhost:2230,localhost:2231,localhost:2232,localhost:2233,localhost:2234
WORKERS="0 1 2 3 4 5 6 7 8 9 10 11"
MODEL_DIR=gs://tomaskrajco-tpu-test-001-bucket/deepxor-async-gd
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=$WORKER_HOSTS \
     --job_name=ps --task_index=0 --model_dir=$MODEL_DIR --train_steps=1000000  &
echo "kill $!" > stop.sh
# On worker0.example.com:
for idx in $WORKERS; do
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=$WORKER_HOSTS \
     --job_name=worker --task_index=$idx \
     --rolls=64 --rolls_len=32 --model_dir=$MODEL_DIR --train_steps=1000000 &
echo "kill $!" >> stop.sh
done
chmod u+x stop.sh
