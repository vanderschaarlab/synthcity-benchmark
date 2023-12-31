import pickle
import os
from pathlib import Path
import argparse

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import synthcity.logger as log

# log.add("synthcity", "DEBUG")

KWARGS = {"n_iter": 100}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def run_dataset(X, workspace_path, model, task_type="regression"):
    loader = GenericDataLoader(X, target_column="y")

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


def run_synthcity(data_type="num", task_type="regression", model="ctgan"):
    file_path = f"./data/{data_type}/{task_type}/"
    workspace_path = Path(f"./workspace/{data_type}/{task_type}/")
    result_path = f"./results/{data_type}/{task_type}/{model}"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # list files in the file_path
    files = os.listdir(file_path)
    print(f"Number of files in {file_path}: {len(files)}")

    for file in files:
        print(f"{file_path}/{file}")
        with open(f"{file_path}/{file}", "rb") as f:
            data_dict = pickle.load(f)

        X = data_dict["X"]
        y = data_dict["y"]
        X["y"] = y

        score = run_dataset(X, workspace_path, model, task_type=task_type)
        if score:
            with open(f"{result_path}/{file}-{model}-{KWARGS_str}.pkl", "wb") as f:
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
