import logging

import numpy as np
from cil_project.utils import masked_rmse

from .baseline import Baseline

logger = logging.getLogger(__name__)


class SVP(Baseline):
    def __init__(self, k: int = 4, max_iter: int = 20, eta: float = 1.3315789473684212, verbose: bool = False):
        super().__init__()  # Initialize the Baseline class
        self.u: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.verbose = verbose
        # Define hyperparameters dictionary
        self.hyperparameters = {"k": k, "max_iter": max_iter, "eta": eta}

    # pylint: disable=too-many-locals
    def svp_matrix_completion(self, d_matrix: np.ndarray) -> None:
        """
        Singular Value Projection for matrix completion.

        :param d_matrix: The normalized and zero-imputed data matrix.
        """

        k = self.hyperparameters["k"]
        max_iter = self.hyperparameters["max_iter"]
        eta = self.hyperparameters["eta"]

        d_matrix_mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        self.reconstructed_matrix = np.zeros((num_users, num_movies))

        for it in range(max_iter):
            self.reconstructed_matrix = self.reconstructed_matrix + eta * d_matrix_mask * (
                d_matrix - self.reconstructed_matrix
            )
            u, s, vt = np.linalg.svd(self.reconstructed_matrix, full_matrices=False)
            # Keep only the top k components
            s_k = np.diag(s[:k])
            u_k = u[:, :k]
            vt_k = vt[:k, :]
            self.reconstructed_matrix = u_k @ s_k @ vt_k
            if self.verbose and not (self.test_m.size == 0 and self.test_m_mask.size == 0):
                r = np.clip(self.reconstructed_matrix.copy() * self.column_std + self.column_mean, 1, 5)
                rmse = masked_rmse(self.test_m, r, self.test_m_mask)
                self.rmse = rmse
                logger.info(f"Epoch {it+1}/{max_iter}, Validation RMSE: {rmse}")

    def train(
        self, data_matrix: np.ndarray, test_m: np.ndarray = np.array([]), test_m_mask: np.ndarray = np.array([])
    ) -> None:
        """
        Training procedure for the SVP model.

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
        self.svp_matrix_completion(data_matrix_norm)
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
