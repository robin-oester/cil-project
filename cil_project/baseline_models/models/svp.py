import numpy as np
from cil_project.ensembling import RatingPredictor

from .baseline import Baseline


class SVP(Baseline, RatingPredictor):
    def __init__(self, k: int = 5, max_iter: int = 60, eta: float = 0.8, verbose: bool = False):
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
        """
        k = self.hyperparameters["k"]
        max_iter = self.hyperparameters["max_iter"]
        eta = self.hyperparameters["eta"]

        d_matrix_mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        self.reconstructed_matrix = np.zeros((num_users, num_movies))

        for it in range(max_iter):
            if self.verbose:
                print(f"Iteration {it+1}")
            self.reconstructed_matrix = self.reconstructed_matrix + eta * d_matrix_mask * (
                d_matrix - self.reconstructed_matrix
            )
            u, s, vt = np.linalg.svd(self.reconstructed_matrix, full_matrices=False)
            # Keep only the top k components
            s_k = np.diag(s[:k])
            u_k = u[:, :k]
            vt_k = vt[:k, :]
            self.reconstructed_matrix = u_k @ s_k @ vt_k
            if self.verbose:
                print("Training RMSE:", self.training_rmse(d_matrix, d_matrix_mask))

    def train(self, data_matrix: np.ndarray) -> None:
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.svp_matrix_completion(data_matrix_norm)
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
