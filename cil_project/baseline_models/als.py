import numpy as np

from .baseline import Baseline


class ALS(Baseline):
    def __init__(self, k: int = 3, max_iter: int = 20, lam: float = 0.1):
        super().__init__()
        self.u: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.hyperparameters = {
            "k": k,
            "max_iter": max_iter,
            "lam": lam,
        }

    # pylint: disable=too-many-locals
    def als_matrix_completion(self, d_matrix: np.ndarray) -> None:
        """
        Alternating Least Squares for matrix completion.
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
            print(f"Iteration {it+1}")
            for i, row_mask in enumerate(d_matrix_mask):
                v_temp = v.T.copy()
                v_temp[row_mask == 0, :] = 0
                matrix = np.dot(v, v_temp) + lam * np.eye(k)
                b = np.dot(v, row_mask * d_matrix[i].T)
                u[i] = np.linalg.solve(matrix, b).T
            self.reconstructed_matrix = np.dot(u, v)  # reconstruct the matrix to calculate RMSE
            print("Training RMSE after optimizing u matrix:", self.training_rmse(d_matrix, d_matrix_mask))

            for j, col_mask in enumerate(d_matrix_mask.T):
                u_temp = u.copy()
                u_temp[col_mask == 0, :] = 0
                matrix = np.dot(u.T, u_temp) + lam * np.eye(k)
                b = np.dot(u.T, col_mask * d_matrix[:, j])
                v[:, j] = np.linalg.solve(matrix, b)
            self.reconstructed_matrix = np.dot(u, v)  # reconstruct the matrix to calculate RMSE
            print("Training RMSE after optimizing v matrix:", self.training_rmse(d_matrix, d_matrix_mask))

        self.u, self.v = u, v
        self.reconstructed_matrix = np.dot(u, v)

    def train(self, data_matrix: np.ndarray) -> None:
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.als_matrix_completion(data_matrix_norm)
        self.denormalize_and_clip_reconstructed_matrix()
        self.save_model_attributes()

    def predict(self, x: tuple[int, int]) -> float:
        if self.u.size == 0 or self.v.size == 0:
            raise ValueError("Model not trained. Please train the model first.")
        user_id, movie_id = x
        return float(self.reconstructed_matrix[user_id, movie_id])
