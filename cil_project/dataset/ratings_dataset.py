import csv
import logging
import pathlib
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from cil_project.utils import DATA_PATH, MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS, nanmean, nanstd
from torch.utils.data import Dataset

from .target_normalization import TargetNormalization

logger = logging.getLogger(__name__)

REGEX_PATTERN = r"r(\d+)_c(\d+)"


class RatingsDataset(Dataset):
    """
    Dataset holding the ratings.
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Initialize the dataset of N samples with inputs and targets.
        Inputs consists of N rows with users and movies. In targets, we can find
        the corresponding ratings. The dataset is stored denormalized.

        :param inputs: array of shape N x 2.
        :param targets: array of shape N x 1.
        """

        assert inputs.shape[1] == 2
        assert inputs.shape[0] == targets.shape[0]
        assert targets.shape[1] == 1

        self._inputs = inputs
        self._targets = targets

        # compute dataset statistics
        data_matrix = self.get_data_matrix(fill_value=np.nan)

        # get mean by user without considering nans
        self._user_means = nanmean(data_matrix, axis=1).reshape(-1, 1)  # shape: (NUM_USERS, 1)
        self._user_stds = nanstd(data_matrix, axis=1).reshape(-1, 1)

        self._movie_means = nanmean(data_matrix, axis=0).reshape(-1, 1)  # shape: (NUM_MOVIES, 1)
        self._movie_stds = nanstd(data_matrix, axis=0).reshape(-1, 1)

        self._target_mean = targets.mean().reshape(-1, 1)  # shape: (num_targets, 1)
        self._target_std = targets.std().reshape(-1, 1)

        # initially no normalization
        self._normalization: Optional[TargetNormalization] = None

        # make dataset compliant to Iterable framework
        self._iter_index = 0

    def set_dataset_statistics(self, dataset: "RatingsDataset") -> None:
        """
        Update the dataset statistics. This can, e.g., used by a validation set to represent the values
        of the training dataset.

        :param dataset: the dataset from which the statistics are taken.
        """

        self._user_means = dataset._user_means
        self._user_stds = dataset._user_stds

        self._user_stds = dataset._user_stds

        self._movie_means = dataset._movie_means
        self._movie_stds = dataset._movie_stds

        self._movie_stds = dataset._movie_stds

        self._target_mean = dataset._target_mean
        self._target_std = dataset._target_std

    def normalize(self, normalization: TargetNormalization) -> None:
        """
        Normalize the targets according to a normalization type.

        :param normalization: the type of normalization.
        """

        assert self._normalization is None, "Dataset should not be normalized at this point"

        mean, std = self._get_normalization_statistics(normalization)

        assert (
            mean.shape == self._targets.shape
        ), f"Shapes of target and mean do not match ({self._targets.shape} vs {mean.shape}"
        assert (
            std.shape == self._targets.shape
        ), f"Shapes of target and std do not match ({self._targets.shape} vs {std.shape}"

        self._targets = np.divide(self._targets - mean, std)
        mean, std = self._get_normalization_statistics(normalization)

        assert (
            mean.shape == self._targets.shape
        ), f"Shapes of target and mean do not match ({self._targets.shape} vs {mean.shape}"
        assert (
            std.shape == self._targets.shape
        ), f"Shapes of target and std do not match ({self._targets.shape} vs {std.shape}"

        self._targets = np.divide(self._targets - mean, std)
        self._normalization = normalization

    def denormalize(self) -> None:
        """
        Denormalizes the targets if they have been normalized.
        """

        if self._normalization is not None:
            mean, std = self._get_normalization_statistics(self._normalization)
            self._targets = np.multiply(self._targets, std) + mean
        if self._normalization is not None:
            mean, std = self._get_normalization_statistics(self._normalization)
            self._targets = np.multiply(self._targets, std) + mean
        self._normalization = None

    def _get_normalization_statistics(self, normalization: TargetNormalization) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean and std arrays for normalization.

        :param normalization: the type of normalization.
        :return: the mean and std arrays for normalization.
        """

        mean = np.empty((len(self), 1), dtype=np.float32)
        std = np.empty((len(self), 1), dtype=np.float32)

        if normalization.value == TargetNormalization.BY_USER.value:
            mean[:] = self._user_means[self._inputs[:, 0]]
            std[:] = self._user_stds[self._inputs[:, 0]]
        elif normalization.value == TargetNormalization.BY_MOVIE.value:
            mean[:] = self._movie_means[self._inputs[:, 1]]
            std[:] = self._movie_stds[self._inputs[:, 1]]
        elif normalization.value == TargetNormalization.BY_TARGET.value:
            mean.fill(self._target_mean)
            std.fill(self._target_std)
        else:
            # this is TargetNormalization.TO_TANH_RANGE
            mean.fill(3.0)
            std.fill(2.0)

        return mean, std

    def get_denormalization_statistics(self, test_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean and std arrays for denormalization.

        :param test_inputs: the inputs to test (shape: N x 2).
        :return: the mean and std arrays that can be used for denormalization.
        """

        assert test_inputs.shape[1] == 2

        mean = np.empty((test_inputs.shape[0], 1), dtype=np.float32)
        std = np.empty((test_inputs.shape[0], 1), dtype=np.float32)

        if self._normalization is None:
            mean.fill(0.0)
            std.fill(1.0)
            return mean, std

        if self._normalization.value == TargetNormalization.BY_USER.value:
            mean[:] = self._user_means[test_inputs[:, 0]]
            std[:] = self._user_stds[test_inputs[:, 0]]
        elif self._normalization.value == TargetNormalization.BY_MOVIE.value:
            mean[:] = self._movie_means[test_inputs[:, 1]]
            std[:] = self._movie_stds[test_inputs[:, 1]]
        elif self._normalization.value == TargetNormalization.BY_TARGET.value:
            mean.fill(self._target_mean)
            std.fill(self._target_std)
        elif self._normalization.value == TargetNormalization.TO_UNIT_RANGE.value:
            mean.fill(1.0)
            std.fill(4.0)
        else:
            # this is TargetNormalization.TO_TANH_RANGE
            mean.fill(3.0)
            std.fill(2.0)

        return mean, std

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> "RatingsDataset":
        """
        Reads the data from a file. The file has a header and each line has the following format:
        r<user>_c<movie>,<rating>. The rating are floats/integers in [MIN_RATING, MAX_RATING].

        :param file_path: path to the data file.
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
        return cls(np.array(inputs), np.array(targets, dtype=np.float32).reshape((-1, 1)))

    def get_split(self, indices: list[int]) -> "RatingsDataset":
        """
        Returns a split of the dataset, given the indices.

        :param indices: the indices of the entries that are kept.
        :return: a subset of the original dataset.
        """
        assert not self.is_normalized(), "Dataset should not be normalized at this point."

        split_inputs = self._inputs[indices]
        split_targets = self._targets[indices]
        return RatingsDataset(split_inputs, split_targets)

    def store(self, name: str) -> None:
        """
        Stores the dataset with the given name in the data folder.
        The dataset should not be normalized.

        :param name: the name of the dataset.
        """

        assert not self.is_normalized(), "Dataset should be stored denormalized."

        np.savez(DATA_PATH / name, inputs=self._inputs, targets=self._targets)

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

        return RatingsDataset(inputs, targets)

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
            ratings = np.full((NUM_USERS, NUM_MOVIES), fill_value, dtype=np.float32)
        else:
            ratings = np.zeros((NUM_USERS, NUM_MOVIES), dtype=np.float32)

        for (user_id, movie_id), rating in zip(self._inputs, self._targets):
            ratings[user_id][movie_id] = rating
        return ratings

    def get_data_matrix_mask(self) -> np.ndarray:
        """
        Returns the mask of the matrix representation. Each non-zero value marks an observed rating.

        :return: the mask (of shape U x M for U users and M movies)
        """

        ratings = np.zeros((NUM_USERS, NUM_MOVIES), dtype=np.float32)

        for user_id, movie_id in self._inputs:
            ratings[user_id][movie_id] = 1
        return ratings

    def get_num_ratings_per_user(self) -> np.ndarray:
        """
        Returns the number of ratings per user.

        :return: the number of ratings per user as N x 1 array.
        """

        return np.bincount(self._inputs[:, 0], minlength=NUM_USERS).reshape(-1, 1)

    def get_num_ratings_per_movie(self) -> np.ndarray:
        """
        Returns the number of ratings per movie.

        :return: the number of ratings per movie as M x 1 array.
        """

        return np.bincount(self._inputs[:, 1], minlength=NUM_MOVIES).reshape(-1, 1)

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

    def get_data_frame(self):
        """
        Returns the dataset as a pandas DataFrame.
        """
        data = self._inputs
        df = pd.DataFrame(data, columns=["user", "movie"])
        return df
