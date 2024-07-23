import logging

import numpy as np
from cil_project.ensembling import RatingPredictor
from cil_project.utils import masked_rmse

from .baseline import Baseline

logger = logging.getLogger(__name__)


class ALS(Baseline, RatingPredictor):
    def __init__(self, k: int = 3, max_iter: int = 21, lam: float = 0.2826666666666667, verbose: bool = False):
        super().__init__()
        self.u: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.verbose = verbose
        # hyperparameters
        self.hyperparameters = {
            "k": k,
            "max_iter": max_iter,
            "lam": lam,
        }

    # pylint: disable=too-many-locals
    def als_matrix_completion(self, d_matrix: np.ndarray) -> None:
        """
        Alternating Least Squares for matrix completion.

        :param d_matrix: The normalized and zero-imputed data matrix.
        """
        # hyperparameters
        k = self.hyperparameters["k"]
        max_iter = self.hyperparameters["max_iter"]
        lam = self.hyperparameters["lam"]

        d_matrix_mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        u = np.random.rand(num_users, k)
        v = np.random.rand(k, num_movies)

        for it in range(max_iter):
            for i, row_mask in enumerate(d_matrix_mask):
                v_temp = v.T.copy()
                v_temp[row_mask == 0, :] = 0
                matrix = np.dot(v, v_temp) + lam * np.eye(k)
                b = np.dot(v, row_mask * d_matrix[i].T)
                u[i] = np.linalg.solve(matrix, b).T

            for j, col_mask in enumerate(d_matrix_mask.T):
                u_temp = u.copy()
                u_temp[col_mask == 0, :] = 0
                matrix = np.dot(u.T, u_temp) + lam * np.eye(k)
                b = np.dot(u.T, col_mask * d_matrix[:, j])
                v[:, j] = np.linalg.solve(matrix, b)

            self.reconstructed_matrix = np.dot(u, v)  # reconstruct the matrix to calculate RMSE
            if self.verbose and not (self.test_m.size == 0 and self.test_m_mask.size == 0):
                r = np.clip(self.reconstructed_matrix.copy() * self.column_std + self.column_mean, 1, 5)
                rmse = masked_rmse(self.test_m, r, self.test_m_mask)
                self.rmse = rmse
                logger.info(f"Epoch {it+1}/{max_iter}, Validation RMSE: {rmse}")

        self.u, self.v = u, v
        self.reconstructed_matrix = np.dot(u, v)

    def train(
        self, data_matrix: np.ndarray, test_m: np.ndarray = np.array([]), test_m_mask: np.ndarray = np.array([])
    ) -> None:
        """
        Training procedure for the ALS model.

        :param data_matrix: The data matrix to train the model on.
        :param test_m: The test data matrix to validate the model on.
        :param test_m_mask: The mask for the test data matrix.
        """
        self.test_m = test_m
        self.test_m_mask = test_m_mask
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.als_matrix_completion(data_matrix_norm)
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
