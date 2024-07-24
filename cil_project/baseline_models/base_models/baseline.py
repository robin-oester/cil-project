from abc import abstractmethod
from typing import Any

import numpy as np
from cil_project.ensembling import RatingPredictor


class Baseline(RatingPredictor):
    """
    Abstract class for all baseline models.
    """

    def __init__(self) -> None:
        self.hyperparameters: dict[str, Any] = {}
        self.reconstructed_matrix: np.ndarray = np.array([])
        self.column_mean: np.ndarray = np.array([])
        self.column_std: np.ndarray = np.array([])
        self.test_m = np.array([])
        self.test_m_mask = np.array([])
        self.rmse = 0.0

    @abstractmethod
    def train(
        self, data_matrix: np.ndarray, test_m: np.ndarray = np.array([]), test_m_mask: np.ndarray = np.array([])
    ) -> None:
        raise NotImplementedError()

    def normalize_data_matrix(self, d_matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes each column of data matrix to have mean 0 and standard deviation 1.

        :param d_matrix: The data matrix to normalize.
        :return: The normalized data matrix.
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
        Denormalizes the reconstructed matrix and clips the values to the range [1, 5].
        """
        self.reconstructed_matrix = self.reconstructed_matrix * self.column_std + self.column_mean
        self.reconstructed_matrix = np.clip(self.reconstructed_matrix, 1, 5)
