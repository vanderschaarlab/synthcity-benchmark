import pickle
import os
from pathlib import Path
import argparse
from arfpy import arf

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
from synthcity.metrics import Metrics
import synthcity.logger as log

# log.add("synthcity", "DEBUG")

KWARGS = {"n_iter": 100}
KWARGS_str = "-".join([f"{k}:{v}" for k, v in KWARGS.items()])


def generate_data(X, delta=0.5):
    try:
        # Train the ARF
        my_arf = arf.arf(x=X, delta=delta)

        # Get density estimates
        my_arf.forde()

        # Generate data
        X_syn = my_arf.forge(n=X.shape[0])
    except Exception as e:
        print(e)
    return X_syn


def run_dataset(X, workspace_path, task_type="regression", delta=0.5):
    try:
        X_syn = generate_data(X, delta=delta)

        evaluation = Metrics.evaluate(
            X,  # DataLoader containing the ground truth dataset
            X_syn,  # DataLoader containing the synthetic dataset
            metrics={
                "stats": ["alpha_precision"],
                "detection": ["detection_xgb", "detection_mlp", "detection_linear"],
                "performance": ["linear_model", "mlp", "xgb"],
            },  # dict where the keys are the metrics types (e.g. "stats", "performance") and the values are the list of metric names (e.g. "alpha_precision", "xgb")
            task_type=task_type,  # "classification", "regression", "survival_analysis", "time_series"
            workspace="workspace",
        )

    except Exception as e:
        print("\n\n", e)
        print(workspace_path, task_type, KWARGS)
        evaluation = None
    return evaluation


def run_synthcity(data_type="num", task_type="regression", delta=0.5):
    file_path = f"./data/{data_type}/{task_type}/"
    workspace_path = Path(f"./workspace/{data_type}/{task_type}/")
    result_path = f"./results/{data_type}/{task_type}/arf"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # list files in the file_path
    files = os.listdir(file_path)
    print(f"Number of files in {file_path}: {len(files)}")

    areas = []
    for file in files:
        with open(f"{file_path}/{file}", "rb") as f:
            data_dict = pickle.load(f)

        X = data_dict["X"]
        y = data_dict["y"]
        X["y"] = y

        areas.append(X.shape[0] * X.shape[1])
    files_2_sort = list(zip(files, areas))
    sorted_files = sorted(files_2_sort, key=lambda x: x[1])

    for file_t in sorted_files:
        file = file_t[0]
        print(f"{file_path}/{file}; shape: {X.shape}")
        with open(f"{file_path}/{file}", "rb") as f:
            data_dict = pickle.load(f)

        X = data_dict["X"]
        y = data_dict["y"]
        X["y"] = y

        score = run_dataset(X, workspace_path, task_type=task_type, delta=delta)
        if score is not None:
            print(score)
            with open(f"{result_path}/{file}-arf-{KWARGS_str}.pkl", "wb") as f:
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
        "--delta",
        type=float,
        default=0.5,
        choices=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0],
    )

    args = parser.parse_args()
    run_synthcity(args.data_type, args.task_type, args.delta)
