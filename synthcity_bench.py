import pickle
import os
from pathlib import Path
import argparse

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks

MODEL = 'ctgan'
KWARGS = {"batch_size": 100, "generator_n_layers_hidden": 2, "discriminator_n_layers_hidden": 2,
          "generator_n_units_hidden": 100, "discriminator_n_units_hidden": 100}


def run_dataset(X, workspace_path, task_type='regression'):
    loader = GenericDataLoader(X, target_column='y')
    score = Benchmarks.evaluate(
        [(MODEL, MODEL, KWARGS)],
        loader.train(),
        loader.test(),
        task_type=task_type,
        synthetic_size=X.shape[0],
        workspace=workspace_path,
        repeats=1,
    )

    return score


def run_synthcity(data_type='num', task_type='regression'):
    file_path = f'./data/{data_type}/{task_type}/'
    workspace_path = Path(f'./workspace/{data_type}/{task_type}/')
    result_path = f'./results/{data_type}/{task_type}/'
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # list files in the file_path
    files = os.listdir(file_path)
    print(f"Number of files in {file_path}: {len(files)}")

    for file in files:
        with open(f'{file_path}/{file}', 'rb') as f:
            data_dict = pickle.load(f)

        X = data_dict['X']
        y = data_dict['y']
        X['y'] = y

        score = run_dataset(X, workspace_path, task_type=task_type)
        with open(f'{result_path}/{file}-{MODEL}.pkl', 'wb') as f:
            pickle.dump(score, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='num', choices=['num', 'cat'])
    parser.add_argument('--task_type', type=str, default='regression', choices=['regression', 'classification'])

    args = parser.parse_args()
    run_synthcity(args.data_type, args.task_type)
