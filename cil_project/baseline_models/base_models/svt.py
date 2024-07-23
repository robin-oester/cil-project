import logging

import numpy as np
from cil_project.ensembling import RatingPredictor
from cil_project.utils import masked_rmse

from .baseline import Baseline

logger = logging.getLogger(__name__)


class SVT(Baseline, RatingPredictor):
    def __init__(self, max_iter: int = 60, eta: float = 1.05, tau: float = 37, verbose: bool = False):
        super().__init__()  # Initialize the Baseline class
        self.u: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.verbose = verbose
        # Define hyperparameters dictionary
        self.hyperparameters = {"max_iter": max_iter, "eta": eta, "tau": tau}

    # pylint: disable=too-many-locals
    def svt_matrix_completion(self, d_matrix: np.ndarray) -> None:
        """
        Singular Value Thresholding for matrix completion.

        :param d_matrix: The normalized and zero-imputed data matrix.
        """
        max_iter = self.hyperparameters["max_iter"]
        eta = self.hyperparameters["eta"]
        tau = self.hyperparameters["tau"]

        mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        self.reconstructed_matrix = np.zeros((num_users, num_movies))

        for it in range(max_iter):
            self.reconstructed_matrix = self.reconstructed_matrix + eta * mask * (d_matrix - self.reconstructed_matrix)
            # perform shrinking
            u, s, vt = np.linalg.svd(self.reconstructed_matrix, full_matrices=False)
            s = (s - tau).clip(min=0)
            nr_selected_sigmas = np.count_nonzero(s)
            self.reconstructed_matrix = (u[:, :nr_selected_sigmas] * s[:nr_selected_sigmas]) @ vt[
                :nr_selected_sigmas, :
            ]
            if self.verbose and not (self.test_m.size == 0 and self.test_m_mask.size == 0):
                r = np.clip(self.reconstructed_matrix.copy() * self.column_std + self.column_mean, 1, 5)
                rmse = masked_rmse(self.test_m, r, self.test_m_mask)
                self.rmse = rmse
                logger.info(f"Epoch {it+1}/{max_iter}, Validation RMSE: {rmse}")

    def train(
        self, data_matrix: np.ndarray, test_m: np.ndarray = np.array([]), test_m_mask: np.ndarray = np.array([])
    ) -> None:
        """
        Training procedure for the SVT model.

        :param data_matrix: The data matrix to train the model on.
        :param test_m: The test data matrix to validate the model on.
        :param test_m_mask: The mask of the test data matrix.
        """
        self.test_m = test_m
        self.test_m_mask = test_m_mask
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.svt_matrix_completion(data_matrix_norm)
        self.denormalize_and_clip_reconstructed_matrix()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict the ratings for the given inputs.

        :param inputs: The inputs to predict the ratings for (shape: (N, 2)).
        :return: The predicted ratings (shape: (N, 1)).
        """
        if self.reconstructed_matrix.size == 0:
            raise ValueError("Model not trained. Please train the model first.")
        users = inputs[:, 0]
        movies = inputs[:, 1]
        estimated_ratings = self.reconstructed_matrix[users, movies]
        return estimated_ratings.reshape(-1, 1)

    def get_name(self) -> str:
        return self.__class__.__name__
