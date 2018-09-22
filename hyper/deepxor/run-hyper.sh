gcloud ml-engine jobs submit training $1 --runtime-version 1.9 --python-version 3.5 --scale-tier basic_gpu --package-path /home/tomaskrajco/tpu-test-001/hyper/deepxor --module-name deepxor.cube --job-dir gs://tomaskrajco-tpu-test-001-bucket/hyper_2 --config=/home/tomaskrajco/tpu-test-001/hyper/deepxor/hyperparam.yaml --region=us-east1 -- --train_steps=47000 --rolls=64 --rolls_len=64 --checkpoint_steps=10000 --model_dir=gs://tomaskrajco-tpu-test-001-bucket/hyper_2 --time_per_move=45 --eval_steps=5000
