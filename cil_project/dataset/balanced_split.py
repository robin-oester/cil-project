from collections import defaultdict

import numpy as np

from .ratings_dataset import RatingsDataset


class BalancedSplit:
    """
    Class used to split a dataset into train and validation set where each user is represented equally.
    """

    def __init__(self, split_percentage: float, shuffle: bool) -> None:
        """
        Initialize the balanced splitter.

        :param split_percentage: percentage of the data that should belong to the training set.
        :param shuffle: whether the data should be shuffled.
        """

        assert 0 <= split_percentage <= 1.0

        self._split_percentage = split_percentage
        self._shuffle = shuffle

    def split(self, dataset: RatingsDataset) -> tuple[list[int], list[int]]:
        """
        Splits the given dataset.

        :param dataset: dataset that is split into train and validation set.
        :return: pair of lists containing the indices for the train and validation set.
        """

        # Organize indices by user id
        user_dict: dict[int, list[int]] = defaultdict(list)  # missing key returns empty list

        for idx, (inputs, _) in enumerate(dataset):
            user_dict[inputs[0].item()].append(idx)

        train_indices = []
        val_indices = []
        for indices in user_dict.values():
            if self._shuffle:
                np.random.shuffle(indices)  # shuffle indices for randomness

            num_train: int = int(len(indices) * self._split_percentage)
            train_indices.extend(indices[:num_train])
            val_indices.extend(indices[num_train:])

        return train_indices, val_indices
