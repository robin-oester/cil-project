import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from cil_project.dataset import RatingsDataset
from cil_project.utils import NUM_MOVIES, NUM_USERS
from myfm import RelationBlock  # pylint: disable=E0401
from scipy import sparse as sps

logger = logging.getLogger(__name__)


class AbstractModel(ABC):
    """
    Abstract bfm model
    """

    def __init__(
        self,
        rank: int = 4,
        grouped: bool = False,
        implicit: bool = False,
        statistical_features: bool = False,
    ) -> None:
        """
        Initializes a new bfm given some model configuration options.

        :param hyperparameters: consists of all model configuration options.
        """

        super().__init__()
        self.rank = rank
        self.grouped = grouped
        self.implicit = implicit
        self.statistical_features = statistical_features
        self.num_bins = 50

    @abstractmethod
    def train(
        self,
        train_dataset: RatingsDataset,
        test_dataset: RatingsDataset,
        n_iter: int = 300,
    ) -> float:
        """
        Trains the model on the given training data and evaluates it on the given test data.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given inputs.
        """
        raise NotImplementedError()

    def augment_user_id(
        self,
        user_vs_watched: dict[int, list[int]],
    ) -> sps.csr_matrix:

        xs = []
        x_uid = sps.eye(NUM_USERS, format="lil")
        xs.append(x_uid)

        if self.implicit:
            x_iu = sps.lil_matrix((NUM_USERS, NUM_MOVIES))
            for index in range(NUM_USERS):
                watched_movies = user_vs_watched.get(index, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for uid in watched_movies:
                    x_iu[index, uid] = normalizer
            xs.append(x_iu)

        return sps.hstack(xs, format="csr")

    def augment_movie_id(
        self,
        movie_vs_watched: dict[int, list[int]],
    ) -> sps.csr_matrix:

        xs = []
        x_movie = sps.eye(NUM_MOVIES, format="lil")
        xs.append(x_movie)

        if self.implicit:
            x_ii = sps.lil_matrix((NUM_MOVIES, NUM_USERS))
            for index in range(NUM_MOVIES):
                watched_users = movie_vs_watched.get(index, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    x_ii[index, uid] = normalizer
            xs.append(x_ii)

        return sps.hstack(xs, format="csr")

    def bin_and_one_hot_encode(self, inputs: np.ndarray, minmax: Tuple[float, float]) -> sps.csr_matrix:
        """
        Bin the inputs into k bins and then one-hot encode the binned values.

        Parameters:
        - inputs (np.ndarray): A 1D numpy array containing the values to bin
        - minmax (Tuple[float, float]): The minimum and maximum values for binning.

        Returns:
        - sp.csr_matrix: A CSR matrix of the one-hot encoded binned values.
        """
        k = self.num_bins
        # Bin the inputs into k bins
        bin_edges = np.linspace(minmax[0], minmax[1], k + 1)
        binned_inputs = np.digitize(inputs, bin_edges, right=False) - 1  # subtract 1 to start bins from 0

        # Create the one-hot encoding in CSR format
        row_indices = np.arange(binned_inputs.size)
        col_indices = binned_inputs
        data = np.ones_like(binned_inputs)

        # Ensure row_indices, col_indices, and data are 1-D
        row_indices = np.ravel(row_indices)
        col_indices = np.ravel(col_indices)
        data = np.ravel(data)

        one_hot_encoded_matrix = sps.csr_matrix((data, (row_indices, col_indices)), shape=(binned_inputs.size, k))

        return one_hot_encoded_matrix

    def get_implicit_features(self, train_dataset: RatingsDataset) -> Tuple[dict[int, list[int]], dict[int, list[int]]]:
        """
        Extracts the implicit features from the training dataset.
        """
        df_train = train_dataset.get_data_frame()

        movie_vs_watched: dict[int, list[int]] = {}
        user_vs_watched: dict[int, list[int]] = {}
        for row in df_train.itertuples():
            user_id = int(row.user)
            movie_id = int(row.movie)
            movie_vs_watched.setdefault(movie_id, []).append(user_id)
            user_vs_watched.setdefault(user_id, []).append(movie_id)

        return user_vs_watched, movie_vs_watched

    def get_group_shapes(self) -> list[int]:
        """
        Returns the group shapes for the model.
        """
        if self.grouped:
            if self.implicit:
                if self.statistical_features:
                    return [
                        self.num_bins,
                        self.num_bins,
                        NUM_USERS,
                        NUM_MOVIES,
                        self.num_bins,
                        self.num_bins,
                        NUM_MOVIES,
                        NUM_USERS,
                    ]
                return [NUM_USERS, NUM_MOVIES, NUM_MOVIES, NUM_USERS]
            if self.statistical_features:
                return [self.num_bins, self.num_bins, NUM_USERS, self.num_bins, self.num_bins, NUM_MOVIES]
            return [NUM_USERS, NUM_MOVIES]

        return None

    # pylint: disable=R0914
    def get_features(
        self,
        dataset: RatingsDataset,
        train_dataset: RatingsDataset,
    ) -> Tuple:
        df = dataset.get_data_frame()

        users = df["user"].to_numpy(dtype=int)
        movies = df["movie"].to_numpy(dtype=int)

        user_vs_watched, movie_vs_watched = self.get_implicit_features(train_dataset)

        user_data = self.augment_user_id(user_vs_watched)
        movie_data = self.augment_movie_id(movie_vs_watched)

        block_user = RelationBlock(users, user_data)
        block_movie = RelationBlock(movies, movie_data)

        mean_minmax = (1, 5 + 1e-6)  # add a small value to the max to include the max value in the last bin
        std_minmax = (0, 4 + 1e-6)

        if not self.statistical_features:
            return [block_user, block_movie]

        # Bin and one-hot encode the user means, standard deviations, and movie means and standard deviations
        ohe_user_means = self.bin_and_one_hot_encode(train_dataset._user_means, mean_minmax)
        ohe_user_stds = self.bin_and_one_hot_encode(train_dataset._user_stds, std_minmax)
        user_stats = RelationBlock(users, sps.hstack([ohe_user_means, ohe_user_stds], format="csr"))

        ohe_movie_means = self.bin_and_one_hot_encode(train_dataset._movie_means, mean_minmax)
        ohe_movie_stds = self.bin_and_one_hot_encode(train_dataset._movie_stds, std_minmax)
        movie_stats = RelationBlock(movies, sps.hstack([ohe_movie_means, ohe_movie_stds], format="csr"))

        return [user_stats, block_user, movie_stats, block_movie]
