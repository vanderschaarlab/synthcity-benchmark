python -u open_ml.py --data_type=num --task_type=regression
python -u open_ml.py --data_type=num --task_type=classification
python -u open_ml.py --data_type=cat --task_type=regression
python -u open_ml.py --data_type=cat --task_type=classification


echo "python -u synthcity_bench_missing_goggle_files.py --data_type=cat --task_type=regression  --model=goggle"
python -u synthcity_bench_missing_goggle_files.py --data_type=cat --task_type=regression  --model=goggle
echo "python -u synthcity_bench_missing_goggle_files.py --data_type=cat --task_type=classification  --model=goggle"
python -u synthcity_bench_missing_goggle_files.py --data_type=cat --task_type=classification  --model=goggle
echo "python -u synthcity_bench_missing_goggle_files.py --data_type=num --task_type=regression  --model=goggle"
python -u synthcity_bench_missing_goggle_files.py --data_type=num --task_type=regression  --model=goggle
echo "python -u synthcity_bench_missing_goggle_files.py --data_type=num --task_type=classification  --model=goggle"
python -u synthcity_bench_missing_goggle_files.py --data_type=num --task_type=classification  --model=goggle
