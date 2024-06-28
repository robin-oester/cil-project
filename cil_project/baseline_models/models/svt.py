import numpy as np
from cil_project.ensembling import RatingPredictor

from .baseline import Baseline


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
        """
        max_iter = self.hyperparameters["max_iter"]
        eta = self.hyperparameters["eta"]
        tau = self.hyperparameters["tau"]

        mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        self.reconstructed_matrix = np.zeros((num_users, num_movies))

        for it in range(max_iter):
            if self.verbose:
                print(f"Iteration {it+1}")
            self.reconstructed_matrix[mask] = self.reconstructed_matrix[mask] + eta * (
                d_matrix[mask] - self.reconstructed_matrix[mask]
            )
            # perform shrinking
            u, s, vt = np.linalg.svd(self.reconstructed_matrix, full_matrices=False)
            s = (s - tau).clip(min=0)
            nr_selected_sigmas = np.count_nonzero(s)
            self.reconstructed_matrix = (u[:, :nr_selected_sigmas] * s[:nr_selected_sigmas]) @ vt[
                :nr_selected_sigmas, :
            ]
            if self.verbose:
                print("Training RMSE:", self.training_rmse(d_matrix, mask))

    def train(self, data_matrix: np.ndarray) -> None:
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.svt_matrix_completion(data_matrix_norm)
        self.denormalize_and_clip_reconstructed_matrix()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.reconstructed_matrix.size == 0:
            raise ValueError("Model not trained. Please train the model first.")
        users = inputs[:, 0]
        movies = inputs[:, 1]
        estimated_ratings = self.reconstructed_matrix[users, movies]
        return estimated_ratings.reshape(-1, 1)

    def get_name(self) -> str:
        return self.__class__.__name__