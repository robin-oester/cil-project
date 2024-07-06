import os
import pathlib
from typing import Union

import numpy as np
import torch
from cil_project.dataset import BalancedKFold, BalancedSplit, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

NR_FOLDS = 10
ENSEMBLER = 0
FOLDER_PATH1 = pathlib.Path(__file__).resolve().parent / "models_kfold_results"
FOLDER_PATH2 = pathlib.Path(__file__).resolve().parent / "models_testset_results"


def init_ensembler(ensembler_id: int) -> Union[LinearRegression, GradientBoostingRegressor]:
    """
    Initialize the ensembler based on the given ID.
    """
    if ensembler_id == 0:
        return LinearRegression()
    if ensembler_id == 1:
        return GradientBoostingRegressor()
    raise ValueError("Invalid ensembler_id")


def read_values_to_array(folder_path: pathlib.Path) -> np.ndarray:
    """
    Read all the values from the text files in the folder and return them as a numpy array.
    Each file's contents are placed in a separate column.
    """

    columns = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                float_values = [float(line.strip()) for line in lines]

                columns.append(np.array(float_values).reshape(-1, 1))

    result_array = np.hstack(columns) if columns else np.array([])

    return result_array


def main() -> None:
    """
    Main function for evaluating the chosen ensembler.
    """

    # prepare the data
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    splitter = BalancedSplit(0.9, True)
    train_idx, test_idx = splitter.split(dataset)
    train_dataset = dataset.get_split(train_idx)
    test_dataset = dataset.get_split(test_idx)

    kfold = BalancedKFold(num_folds=NR_FOLDS, shuffle=True)

    # Prepare X_train and y_train to train the ensembler
    # pylint: disable=invalid-name
    X_train = read_values_to_array(FOLDER_PATH1)
    y_train = []

    for _, test_idx in kfold.split(train_dataset):
        y_train.append(train_dataset.get_split(test_idx).get_targets())

    y_train = np.concatenate(y_train).reshape(-1)

    # Prepare X_test and y_test to evaluate the ensembler
    X_test = read_values_to_array(FOLDER_PATH2)
    y_test = test_dataset.get_targets().reshape(-1)

    # Train the ensembler
    ensembler = init_ensembler(ENSEMBLER)
    ensembler.fit(X_train, y_train)

    # Evaluate the ensembler
    y_pred = ensembler.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("--------------------------------")
    print(f"Validation RMSE: {rmse}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
