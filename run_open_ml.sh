python -u open_ml.py --data_type=num --task_type=regression
python -u open_ml.py --data_type=num --task_type=classification
python -u open_ml.py --data_type=cat --task_type=regression
python -u open_ml.py --data_type=cat --task_type=classification


echo "python -u synthcity_bench.py --data_type=cat --task_type=regression"
python -u synthcity_bench.py --data_type=cat --task_type=regression
echo "python -u synthcity_bench.py --data_type=cat --task_type=classification"
python -u synthcity_bench.py --data_type=cat --task_type=classification
echo "python -u synthcity_bench.py --data_type=num --task_type=regression"
python -u synthcity_bench.py --data_type=num --task_type=regression
echo "python -u synthcity_bench.py --data_type=num --task_type=classification"
python -u synthcity_bench.py --data_type=num --task_type=classification
