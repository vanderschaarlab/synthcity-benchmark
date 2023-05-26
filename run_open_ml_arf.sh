python -u open_ml.py --data_type=num --task_type=regression
python -u open_ml.py --data_type=cat --task_type=regression
python -u open_ml.py --data_type=num --task_type=classification
python -u open_ml.py --data_type=cat --task_type=classification


echo "python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0.5"
python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0.5
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0.5"
python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0.5
echo "python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0.5"
python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0.5
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0.5"
python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0.5

echo "python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0.3"
python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0.3
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0.3"
python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0.3
echo "python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0.3"
python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0.3
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0.3"
python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0.3

echo "python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0"
python -u synthcity_bench_arf.py --data_type=num --task_type=regression  --delta=0
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0"
python -u synthcity_bench_arf.py --data_type=cat --task_type=classification  --delta=0
echo "python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0"
python -u synthcity_bench_arf.py --data_type=num --task_type=classification  --delta=0
echo "python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0"
python -u synthcity_bench_arf.py --data_type=cat --task_type=regression  --delta=0
