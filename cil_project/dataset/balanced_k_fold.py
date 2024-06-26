from collections import defaultdict
from typing import Iterator

import numpy as np

from .ratings_dataset import RatingsDataset


class BalancedKFold:
    """
    Allows balanced K-folding (each user is approx. equally present in each fold).
    """

    def __init__(self, num_folds: int, shuffle: bool):
        """
        Initializes the BalancedKFold class.

        :param num_folds: the amount of folds.
        :param shuffle: whether data should also be shuffled.
        """

        self._num_folds = num_folds
        self._shuffle = shuffle

    # pylint: disable=too-many-locals
    def split(self, dataset: RatingsDataset) -> Iterator[tuple[list[int], list[int]]]:
        """
        Splits the dataset into the K folds.

        :param dataset: dataset that needs to be split.
        :return: the folds in iterative manner.
        """

        # Organize indices by user id
        user_dict: dict[int, list[int]] = defaultdict(list)  # missing key returns empty list

        for idx, (inputs, _) in enumerate(dataset):
            user_dict[inputs[0].item()].append(idx)

        folds: list[list[int]] = [[] for _ in range(self._num_folds)]

        for indices in user_dict.values():
            if self._shuffle:
                np.random.shuffle(indices)  # shuffle indices for randomness
            fold_size = len(indices) // self._num_folds
            remainder = len(indices) % self._num_folds

            start_idx = 0
            for i in range(self._num_folds):
                end_idx = start_idx + fold_size + (1 if i < remainder else 0)
                folds[i].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        for i in range(self._num_folds):
            train_idx: list[int] = []
            test_idx: list[int] = []
            for j in range(self._num_folds):
                if i == j:
                    test_idx.extend(folds[j])
                else:
                    train_idx.extend(folds[j])
            yield train_idx, test_idx
