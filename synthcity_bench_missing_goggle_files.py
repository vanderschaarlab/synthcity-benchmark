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
        print(model, task_type, KWARGS)
        score = None

    return score


def run_synthcity():
    data_types = ["num", "cat"]
    task_types = ["regression", "classification"]
    model = "goggle"

    # list files in the file_path
    files = [
        "./data/num/regression//42729.pkl",
        "./data/num/regression//43174.pkl",
        "./data/num/regression//300.pkl",
        "./data/num/classification//993.pkl",
        "./data/num/classification//293.pkl",
        "./data/num/classification//41150.pkl",
        "./data/num/classification//42769.pkl",
        "./data/num/classification//1461.pkl",
        "./data/num/classification//354.pkl",
        "./data/cat/regression//42712.pkl",
        "./data/cat/regression//42207.pkl",
        "./data/cat/regression//42571.pkl",
        "./data/cat/regression//42729.pkl",
        "./data/cat/regression//42225.pkl",
        "./data/cat/regression//41540.pkl",
        "./data/cat/regression//43144.pkl",
        "./data/cat/regression//416.pkl",
        "./data/cat/regression//6331.pkl",
        "./data/cat/regression//42731.pkl",
        "./data/cat/regression//42688.pkl",
        "./data/cat/regression//42570.pkl",
        "./data/cat/classification//1596.pkl",
        "./data/cat/classification//1114.pkl",
        "./data/cat/classification//41160.pkl",
    ]
    areas = []
    for file in files:
        with open(f"{file}", "rb") as f:
            data_dict = pickle.load(f)

        X = data_dict["X"]
        y = data_dict["y"]
        X["y"] = y

        areas.append(X.shape[0] * X.shape[1])
    files_2_sort = list(zip(files, areas))
    sorted_files = sorted(files_2_sort, key=lambda x: x[1])

    for file_t in sorted_files:
        file = file_t[0]
        print(f"Running {file}")
        task_type = file.split("/")[3]
        data_type = file.split("/")[2]
        result_path = f"./results/{data_type}/{task_type}/{model}"
        if X.isnull().values.any():
            X.replace(np.NaN, X.median(numeric_only=True), inplace=True)
        results_file = file.split("/")[-1].split(".")[0]
        print(
            f"{result_path}/{results_file}-{model}-{KWARGS_str}.pkl",
        )
        score = run_dataset(X, "workspace", model, task_type=task_type)
        if score:
            with open(
                f"{result_path}/{file}-{model}-{KWARGS_str}.pkl",
                "wb",
            ) as f:
                pickle.dump(score, f)


if __name__ == "__main__":
    run_synthcity()
