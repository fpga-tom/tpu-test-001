# On ps0.example.com:
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
     --job_name=ps --task_index=0 &
# On worker0.example.com:
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
     --job_name=worker --task_index=0 \
     --rolls=64 --rolls_len=32 &
# On worker1.example.com:
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
     --job_name=worker --task_index=1 \
     --rolls=64 --rolls_len=32 &
python3 cube.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224,localhost:2225 \
     --job_name=worker --task_index=2 \
     --rolls=64 --rolls_len=32 &
