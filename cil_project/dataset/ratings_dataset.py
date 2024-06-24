import csv
import logging
import pathlib
import re
from typing import Optional

import numpy as np
import torch
from cil_project.utils import DATA_PATH, MAX_RATING, MIN_RATING
from torch.utils.data import Dataset

from .target_normalization import TargetNormalization

logger = logging.getLogger(__name__)

REGEX_PATTERN = r"r(\d+)_c(\d+)"


class RatingsDataset(Dataset):
    """
    Dataset holding the ratings.
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, num_users: int, num_movies: int) -> None:
        """
        Initialize the dataset of N samples with inputs and targets.
        Inputs consists of N rows with users and movies. In targets, we can find
        the corresponding ratings. The dataset is stored denormalized.

        :param inputs: array of shape N x 2.
        :param targets: array of shape N x 1.
        :param num_users: total number of users.
        :param num_movies: total number of movies.
        """

        assert inputs.shape[1] == 2
        assert inputs.shape[0] == targets.shape[0]
        assert targets.shape[1] == 1

        self._inputs = inputs
        self._targets = targets

        self._num_users = num_users
        self._num_movies = num_movies

        # compute dataset statistics
        data_matrix = self.get_data_matrix()
        self._user_means = data_matrix.mean(1)  # shape: (num_users,)
        self._movie_means = data_matrix.mean(0)  # shape: (num_movies,)
        self._target_mean = targets.mean()

        self._user_stds = data_matrix.std(1)
        self._movie_stds = data_matrix.std(0)
        self._target_std = targets.std()

        # initially no normalization
        self._normalization: Optional[TargetNormalization] = None

        # make dataset compliant to Iterable framework
        self._iter_index = 0

    def set_dataset_statistics(self, dataset: "RatingsDataset") -> None:
        """
        Update the dataset statistics. This can, e.g., used by a test set to represent the values
        of the training dataset.

        :param dataset: the dataset from which the statistics are taken.
        """

        assert self._num_users == dataset._num_users, "User number doesn't match"
        assert self._num_movies == dataset._num_movies, "Movie number doesn't match"

        self._user_means = dataset._user_means
        self._movie_means = dataset._movie_means
        self._target_mean = dataset._target_mean

        self._user_stds = dataset._user_stds
        self._movie_stds = dataset._movie_stds
        self._target_std = dataset._target_std

    def normalize(self, normalization: TargetNormalization) -> None:
        """
        Normalize the targets according to a normalization type.

        :param normalization: the type of normalization.
        """

        assert self._normalization is None, "Dataset should not be normalized at this point"

        # this is Normalization.TO_TANH_RANGE
        mean = 3.0
        std = 2.0
        for i in range(len(self)):
            if normalization == TargetNormalization.BY_USER:
                user_id = self._inputs[i, 0]
                mean = self._user_means[user_id]
                std = self._user_stds[user_id]
            elif normalization == TargetNormalization.BY_MOVIE:
                movie_id = self._inputs[i, 1]
                mean = self._movie_means[movie_id]
                std = self._movie_stds[movie_id]
            elif normalization == TargetNormalization.BY_TARGET:
                mean = self._target_mean
                std = self._target_std

            self._targets[i] = (self._targets[i] - mean) / std

        self._normalization = normalization

    def denormalize(self) -> None:
        """
        Denormalizes the targets if they have been normalized.
        """

        if self.is_normalized():
            # this is Normalization.TO_TANH_RANGE
            mean = 3.0
            std = 2.0

            for i in range(len(self)):
                if self._normalization == TargetNormalization.BY_USER:
                    user_id = self._inputs[i, 0]
                    mean = self._user_means[user_id]
                    std = self._user_stds[user_id]
                elif self._normalization == TargetNormalization.BY_MOVIE:
                    movie_id = self._inputs[i, 1]
                    mean = self._movie_means[movie_id]
                    std = self._movie_stds[movie_id]
                elif self._normalization == TargetNormalization.BY_TARGET:
                    mean = self._target_mean
                    std = self._target_std

                self._targets[i] = self._targets[i] * std + mean
        self._normalization = None

    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Denormalizes the predictions if they have been normalized.

        :param predictions: the predictions to be denormalized.
        :return: the denormalized predictions.
        """

        if self.is_normalized():
            # this is Normalization.TO_TANH_RANGE
            mean = 3.0
            std = 2.0

            for i, _ in enumerate(predictions):
                if self._normalization == TargetNormalization.BY_USER:
                    user_id = self._inputs[i, 0]
                    mean = self._user_means[user_id]
                    std = self._user_stds[user_id]
                elif self._normalization == TargetNormalization.BY_MOVIE:
                    movie_id = self._inputs[i, 1]
                    mean = self._movie_means[movie_id]
                    std = self._movie_stds[movie_id]
                elif self._normalization == TargetNormalization.BY_TARGET:
                    mean = self._target_mean
                    std = self._target_std

                predictions[i] = predictions[i] * std + mean
        return predictions

    @classmethod
    def from_file(cls, file_path: pathlib.Path, num_users: int = 10000, num_movies: int = 1000) -> "RatingsDataset":
        """
        Reads the data from a file. The file has a header and each line has the following format:
        r<user>_c<movie>,<rating>. The rating are floats/integers in [MIN_RATING, MAX_RATING].

        :param file_path: path to the data file.
        :param num_users: total number of users.
        :param num_movies: total number of movies.
        :return: the loaded dataset.
        """

        inputs: list[np.ndarray] = []
        targets: list[float] = []

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                id_str, rating_str = row
                match = re.match(REGEX_PATTERN, id_str)
                if match:
                    # both are 1-based
                    user_idx = int(match.group(1)) - 1
                    movie_idx = int(match.group(2)) - 1
                else:
                    raise ValueError(f"Id '{id_str}' does not match the expected pattern.")
                rating = float(rating_str.strip())

                assert MIN_RATING <= rating <= MAX_RATING, "Rating must be between 1-5"

                inputs.append(np.array([user_idx, movie_idx]))
                targets.append(rating)

        logging.info(f"Loaded a total of {len(targets)} entries.")

        # reshape label to have dim (N, 1) not (N,)
        return cls(np.array(inputs), np.array(targets, dtype=np.float32).reshape((-1, 1)), num_users, num_movies)

    def get_split(self, indices: list[int]) -> "RatingsDataset":
        """
        Returns a split of the dataset, given the indices.

        :param indices: the indices of the entries that are kept.
        :return: a subset of the original dataset.
        """
        assert not self.is_normalized(), "Dataset should not be normalized at this point."

        split_inputs = self._inputs[indices]
        split_targets = self._targets[indices]
        return RatingsDataset(split_inputs, split_targets, self._num_users, self._num_movies)

    def store(self, name: str) -> None:
        """
        Stores the dataset with the given name in the data folder.
        The dataset should not be normalized.

        :param name: the name of the dataset.
        """

        assert not self.is_normalized(), "Dataset should be stored denormalized."

        np.savez(
            DATA_PATH / name,
            inputs=self._inputs,
            targets=self._targets,
            num_users=np.array([self._num_users]),
            num_movies=np.array([self._num_movies]),
        )

    @staticmethod
    def load(name: str) -> "RatingsDataset":
        """
        Loads the dataset with the given name from the data folder.

        :param name: the name of the dataset.
        :return: the loaded dataset.
        """
        path = DATA_PATH / f"{name}.npz"

        assert path.is_file(), "Path should point to a file."

        data = np.load(path)
        inputs = data["inputs"]
        targets = data["targets"]
        num_users = data["num_users"][0]
        num_movies = data["num_movies"][0]

        return RatingsDataset(inputs, targets, num_users, num_movies)

    @staticmethod
    def get_available_dataset_names() -> list[str]:
        """
        Get the names of the available datasets in the data directory.

        :return: the names of the available datasets.
        """

        possible_base_datasets: list[str] = []
        for file in DATA_PATH.iterdir():
            if file.is_file() and file.suffix == ".npz":
                possible_base_datasets.append(file.stem)

        return possible_base_datasets

    def get_data_matrix(self, fill_value: Optional[float] = None) -> np.ndarray:
        """
        Returns the dataset as a matrix. Each entry (u, m) contains the rating of user u for movie m.
        Overrides ratings if there are duplicate (u, m) pairs with the latest one.

        :param fill_value: the value to fill the zero entries with.
        :return: the matrix containing the ratings (of shape U x M for U users and M movies)
        """

        if fill_value is not None:
            ratings = np.full((self._num_users, self._num_movies), fill_value, dtype=np.float32)
        else:
            ratings = np.zeros((self._num_users, self._num_movies), dtype=np.float32)

        for (user_id, movie_id), rating in zip(self._inputs, self._targets):
            ratings[user_id][movie_id] = rating
        return ratings

    def get_data_matrix_mask(self) -> np.ndarray:
        """
        Returns the mask of the matrix representation. Each non-zero value marks an observed rating.

        :return: the mask (of shape U x M for U users and M movies)
        """

        ratings = np.zeros((self._num_users, self._num_movies), dtype=np.float32)

        for user_id, movie_id in self._inputs:
            ratings[user_id][movie_id] = 1
        return ratings

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self._inputs[idx]), torch.from_numpy(self._targets[idx])

    def __iter__(self) -> "RatingsDataset":
        self._iter_index = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._iter_index < len(self):
            result = self.__getitem__(self._iter_index)
            self._iter_index += 1
            return result
        raise StopIteration()

    def is_normalized(self) -> bool:
        return self._normalization is not None

    def get_target_mean(self) -> float:
        return self._targets.mean()

    def get_inputs(self) -> np.ndarray:
        return self._inputs

    def get_targets(self) -> np.ndarray:
        return self._targets
