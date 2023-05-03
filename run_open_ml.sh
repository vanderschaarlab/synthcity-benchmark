python -u open_ml.py --data_type=num --task_type=regression
python -u open_ml.py --data_type=num --task_type=classification
python -u open_ml.py --data_type=cat --task_type=regression
python -u open_ml.py --data_type=cat --task_type=classification


python -u synthcity_bench.py --data_type=num --task_type=regression --model=ctgan
python -u synthcity_bench.py --data_type=num --task_type=classification  --model=ctgan
python -u synthcity_bench.py --data_type=cat --task_type=regression  --model=ctgan
python -u synthcity_bench.py --data_type=cat --task_type=classification  --model=ctgan

python -u synthcity_bench.py --data_type=num --task_type=regression --model=tvae
python -u synthcity_bench.py --data_type=num --task_type=classification  --model=tvae
python -u synthcity_bench.py --data_type=cat --task_type=regression  --model=tvae
python -u synthcity_bench.py --data_type=cat --task_type=classification  --model=tvae
