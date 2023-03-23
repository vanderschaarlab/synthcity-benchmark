python -u open_ml.py --data_type=num --task_type=regression &
python -u open_ml.py --data_type=num --task_type=classification &
python -u open_ml.py --data_type=cat --task_type=regression &
python -u open_ml.py --data_type=cat --task_type=classification &


nohup python -u synthcity_bench.py --data_type=num --task_type=regression &
nohup python -u synthcity_bench.py --data_type=num --task_type=classification &
nohup python -u synthcity_bench.py --data_type=cat --task_type=regression &
nohup python -u synthcity_bench.py --data_type=cat --task_type=classification &
