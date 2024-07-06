import os
import pathlib
from typing import Union

import numpy as np
import torch
from cil_project.dataset import BalancedKFold, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

NR_FOLDS = 10
ENSEMBLER = 0
FOLDER_PATH = pathlib.Path(__file__).resolve().parent / "models_kfold_results"


def init_ensembler(ensembler_id: int) -> Union[LinearRegression, GradientBoostingRegressor]:
    """
    Initialize the ensembler based on the given ID.
    """
    if ensembler_id == 0:
        return LinearRegression()
    if ensembler_id == 1:
        return GradientBoostingRegressor()
    raise ValueError("Invalid ensembler_id")


def read_values_to_array(folder_path: pathlib.Path, start_index: int, end_index: int) -> np.ndarray:
    """
    Read the values from the text files in the given folder and return them as a numpy array.
    The values are read from the lines between start_index and end_index.
    """
    columns = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[start_index:end_index]
                float_values = [float(line.strip()) for line in lines]

                columns.append(np.array(float_values).reshape(-1, 1))

    result_array = np.hstack(columns) if columns else np.array([])

    return result_array


# pylint: disable=too-many-locals


def main() -> None:
    """
    Main function for evaluating the chosen ensembler.
    """

    # prepare the data
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    kfold = BalancedKFold(num_folds=NR_FOLDS, shuffle=True)
    rmse_scores = []

    start_idx = 0
    for fold, (_, test_idx) in enumerate(kfold.split(dataset)):
        end_idx = start_idx + len(test_idx)
        # pylint: disable=invalid-name
        X = read_values_to_array(FOLDER_PATH, start_idx, end_idx)
        y = dataset.get_split(test_idx).get_targets().reshape(-1)

        # train ensemler on one half of validation fold and validate on the other half and vice versa
        kf = KFold(n_splits=2, shuffle=True)
        for split, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{NR_FOLDS}, Split {split + 1}/2. ", end="")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ensembler = init_ensembler(ENSEMBLER)
            ensembler.fit(X_train, y_train)

            y_pred = ensembler.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
            print(f"Validation RMSE: {rmse}")

        start_idx = end_idx

    average_rmse = np.mean(rmse_scores)
    print("--------------------------------")
    print(f"Average RMSE: {average_rmse}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
