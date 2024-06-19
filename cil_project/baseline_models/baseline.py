import pathlib
import pickle
import time
from typing import Any

import numpy as np


class Baseline:
    def __init__(self) -> None:
        self.hyperparameters: dict[str, Any] = {}
        self.reconstructed_matrix: np.ndarray = np.array([])
        self.column_mean: np.ndarray = np.array([])
        self.column_std: np.ndarray = np.array([])

    def normalize_data_matrix(self, d_matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes each column of data matrix to have mean 0 and standard deviation 1.
        """
        col_mu = np.nanmean(d_matrix, axis=0)
        col_sigma = np.nanstd(d_matrix, axis=0)
        d_matrix_norm = (d_matrix - col_mu) / col_sigma
        # store the mean and std for each column
        self.column_mean = col_mu
        self.column_std = col_sigma
        # return the normalized matrix
        return d_matrix_norm

    def denormalize_and_clip_reconstructed_matrix(self) -> None:
        """
        Denormalizes the reconstruvted matrix and clips the values to the range [1, 5].
        """
        self.reconstructed_matrix = self.reconstructed_matrix * self.column_std + self.column_mean
        self.reconstructed_matrix = np.clip(self.reconstructed_matrix, 1, 5)

    def training_rmse(self, d: np.ndarray, d_mask: np.ndarray) -> float:
        return np.sqrt(np.sum((d_mask * (d - self.reconstructed_matrix)) ** 2) / float(np.sum(d_mask)))

    def save_model_attributes(self) -> None:
        folder_path = pathlib.Path(pathlib.Path(__file__).parent) / "predictor_attributes"
        name = self.__class__.__name__
        folder_path.mkdir(exist_ok=True)
        millis = int(time.time())
        hyperparameters_str = "-".join([f"{key}{value}" for key, value in self.hyperparameters.items()])
        file_name = f"{name}-{millis}-{hyperparameters_str}.pkl"
        file_path = folder_path / file_name
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model_attributes(file_path: pathlib.Path) -> None:
        """Loads and returns a model instance from a file."""
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
