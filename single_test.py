import pickle
import os
from pathlib import Path
import argparse

import torch

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import synthcity.logger as log

log.add("synthcity", "DEBUG")

KWARGS = {
    "n_iter": 100,
    "batch_size": 32,
    # "generator_n_layers_hidden": 2,
    # "discriminator_n_layers_hidden": 2,
    # "generator_n_units_hidden": 100,
    # "discriminator_n_units_hidden": 100,
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
            batch_size=KWARGS["batch_size"],
        )
    except Exception as e:
        print("\n\n", e)
        print(workspace_path, model, task_type, KWARGS)
        score = None

    return score


def run_synthcity(data_type="cat", task_type="regression", model="goggle"):
    cwd = Path.cwd()
    file_path = cwd / f"data/{data_type}/{task_type}/"
    workspace_path = cwd / Path(f"workspace/{data_type}/{task_type}/")
    result_path = cwd / f"results/{data_type}/{task_type}/"
    Path(result_path).mkdir(parents=True, exist_ok=True)
    file = "287.pkl"

    print(f"{file_path}/{file}")
    with open(f"{file_path}/{file}", "rb") as f:
        data_dict = pickle.load(f)

    X = data_dict["X"]
    y = data_dict["y"]
    X["y"] = y
    if not "{result_path}/{file}-{model}-{KWARGS_str}.pkl" in os.listdir(result_path):
        score = run_dataset(X, workspace_path, model, task_type=task_type)
        if score:
            with open(
                f"{result_path}/{model}/{file}-{model}-{KWARGS_str}.pkl", "wb"
            ) as f:
                pickle.dump(score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="num", choices=["num", "cat"])
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ctgan",
        choices=["ctgan", "ddpm", "tvae", "goggle"],
    )

    args = parser.parse_args()
    run_synthcity(args.data_type, args.task_type, args.model)
