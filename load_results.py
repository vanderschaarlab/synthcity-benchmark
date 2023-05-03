import os
import pickle
from pathlib import Path

data_types = ["num", "cat"]
task_types = ["regression", "classification"]
model = "ctgan"
KWARGS = {}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])

for data_type in data_types:
    for task_type in task_types:
        result_path = f"./results/{data_type}/{task_type}/"
        files = os.listdir(result_path)
        for file in files:
            print(f"{data_type}/{task_type}/{file}")
            with open(f"./results/{data_type}/{task_type}/{file}", "rb") as f:
                results = pickle.load(f)

                print(results)
