import pickle
import os
from pathlib import Path
import argparse

import torch
import numpy as np

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import synthcity.logger as log

# log.add("synthcity", "DEBUG")

KWARGS = {
    "n_iter": 100,
}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def run_dataset(X, workspace_path, model, task_type="regression"):
    loader = GenericDataLoader(X, target_column="y")
    torch.cuda.empty_cache()
    try:
        score = Benchmarks.evaluate(
            [(model, model, KWARGS)],
            loader.train(),
            loader.test(),
            task_type=task_type,
            synthetic_size=X.shape[0],
            metrics={
                "stats": ["alpha_precision"],
                "detection": ["detection_xgb", "detection_mlp", "detection_linear"],
                "performance": ["linear_model", "mlp", "xgb"],
            },
            workspace=workspace_path,
            repeats=1,
        )
    except Exception as e:
        print("\n\n", e)
        print(workspace_path, model, task_type, KWARGS)
        score = None

    return score


def run_synthcity():
    data_types = ["num", "cat"]
    task_types = ["regression", "classification"]
    model = "goggle"

    for data_type in data_types:
        print(f"Running {data_type} files:")
        for task_type in task_types:
            print(f"Running {task_type} files:")
            file_path = f"./data/{data_type}/{task_type}/"
            result_path = f"./results/{data_type}/{task_type}/"
            # list files in the file_path
            files = os.listdir(file_path)
            for file in files:
                completed_file_data = [
                    (f.split("-")[0].split(".")[0], f.split("-")[1])
                    for f in os.listdir(f"{result_path}/{model}/")
                ]
                file_num = file.split(".")[0]
                if (file_num, model) not in completed_file_data:
                    print(f"Running {file}-{model}")
                    with open(f"{file_path}/{file}", "rb") as f:
                        data_dict = pickle.load(f)

                    X = data_dict["X"]
                    y = data_dict["y"]
                    X["y"] = y

                    if X.isnull().values.any():
                        X.replace(np.NaN, X.median(numeric_only=True), inplace=True)

                    score = run_dataset(X, "workspace", model, task_type=task_type)
                    if score:
                        with open(
                            f"{result_path}/{file}-{model}-{KWARGS_str}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(score, f)


if __name__ == "__main__":
    run_synthcity()
