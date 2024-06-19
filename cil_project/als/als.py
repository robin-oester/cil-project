import pathlib
import pickle
import time

import numpy as np


class ALS:
    def __init__(self, k: int = 3, max_iter: int = 20, lam: float = 0.1):
        self.k = k
        self.max_iter = max_iter
        self.lam = lam
        self.reconstructed_matrix: np.ndarray = np.array([])
        self.u: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.column_mean: np.ndarray = np.array([])
        self.column_std: np.ndarray = np.array([])
        self.load_model_attributes()

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
        self.reconstructed_matrix = (self.u @ self.v) * self.column_std + self.column_mean
        self.reconstructed_matrix = np.clip(self.reconstructed_matrix, 1, 5)

    def training_rmse(self, d: np.ndarray, d_mask: np.ndarray, u_matrix: np.ndarray, v_matrix: np.ndarray) -> float:
        return np.sqrt(np.sum((d_mask * (d - np.dot(u_matrix, v_matrix))) ** 2) / float(np.sum(d_mask)))

    # pylint: disable=too-many-locals
    def als_matrix_completion(self, d_matrix: np.ndarray) -> None:
        """
        Alternating Least Squares for matrix completion.
        """
        d_matrix_mask = d_matrix != 0
        num_users, num_movies = d_matrix.shape
        u = np.random.rand(num_users, self.k)
        v = np.random.rand(self.k, num_movies)

        for it in range(self.max_iter):
            print(f"Iteration {it+1}")
            for i, row_mask in enumerate(d_matrix_mask):
                v_temp = v.T.copy()
                v_temp[row_mask == 0, :] = 0
                matrix = np.dot(v, v_temp) + self.lam * np.eye(self.k)
                b = np.dot(v, row_mask * d_matrix[i].T)
                u[i] = np.linalg.solve(matrix, b).T
            print("Training RMSE after optimizing u matrix:", self.training_rmse(d_matrix, d_matrix_mask, u, v))

            for j, col_mask in enumerate(d_matrix_mask.T):
                u_temp = u.copy()
                u_temp[col_mask == 0, :] = 0
                matrix = np.dot(u.T, u_temp) + self.lam * np.eye(self.k)
                b = np.dot(u.T, col_mask * d_matrix[:, j])
                v[:, j] = np.linalg.solve(matrix, b)
            print("Training RMSE after optimizing v matrix:", self.training_rmse(d_matrix, d_matrix_mask, u, v))

        self.u, self.v = u, v

    def train(self, data_matrix: np.ndarray) -> None:
        if not np.isnan(data_matrix).any():  # If the matrix has already been zero-imputed
            data_matrix[data_matrix == 0] = np.nan
        data_matrix_norm = self.normalize_data_matrix(data_matrix)
        # set nan values to 0 in data_matrix_norm
        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        self.als_matrix_completion(data_matrix_norm)
        self.denormalize_and_clip_reconstructed_matrix()
        self.save_model_attributes()

    def save_model_attributes(self) -> None:
        folder_path = pathlib.Path(pathlib.Path(__file__).parent) / "predictor_attributes"
        folder_path.mkdir(exist_ok=True)
        # Open a file in binary write mode using Path object
        millis = int(time.time())
        with open(
            folder_path / f"attributes_{millis}_k{self.k}_maxiter{self.max_iter}_lam{self.lam}.pkl", "wb"
        ) as file:
            print(f"Saving attributes to file: attributes_{millis}_k{self.k}_maxiter{self.max_iter}_lam{self.lam}.pkl")
            pickle.dump(
                {"u": self.u, "v": self.v, "column_mean": self.column_mean, "column_std": self.column_std}, file
            )

    def load_model_attributes(self) -> None:
        try:
            folder_path = pathlib.Path(pathlib.Path(__file__).parent) / "predictor_attributes"
            # Get all attribute files in the directory
            files = list(folder_path.glob("attributes_*.pkl"))
            # Sort files by their timestamp in the filename, selecting the newest
            latest_file = max(files, key=lambda x: int(x.stem.split("_")[1]))
            with open(latest_file, "rb") as file:
                print(f"Loading attributes of latest file: {latest_file.name}")
                attributes = pickle.load(file)
                self.u = attributes["u"]
                self.v = attributes["v"]
                self.column_mean = attributes["column_mean"]
                self.column_std = attributes["column_std"]
                self.denormalize_and_clip_reconstructed_matrix()
        except ValueError:
            print("Model attributes file not found. Please train the model first.")
        except FileNotFoundError:
            print("Specific model attributes file not found. Please check the file path.")

    def predict(self, x: tuple[int, int]) -> float:
        if self.u.size == 0 or self.v.size == 0:
            raise ValueError("Model not trained. Please train the model first.")
        user_id, movie_id = x
        return float(self.reconstructed_matrix[user_id, movie_id])
