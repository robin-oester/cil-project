from collections import defaultdict
from typing import Iterator

import numpy as np

from .ratings_dataset import RatingsDataset


class BalancedKFold:
    """
    Allows balanced K-folding (each user is present approx. equally in each fold).
    """

    def __init__(self, num_folds: int, shuffle: bool):
        self.num_folds = num_folds
        self.shuffle = shuffle

    def split(self, dataset: RatingsDataset) -> Iterator[tuple[list[int], list[int]]]:
        user_dict: dict[int, list[int]] = defaultdict(list)  # missing key returns empty list

        # Organize indices by user id
        for idx, ((user_id, _), _) in enumerate(dataset):
            user_dict[user_id].append(idx)

        folds: list[list[int]] = [[] for _ in range(self.num_folds)]

        for user_id, indices in user_dict.items():
            if self.shuffle:
                np.random.shuffle(indices)  # Shuffle indices for randomness
            fold_size = len(indices) // self.num_folds
            remainder = len(indices) % self.num_folds

            start_idx = 0
            for i in range(self.num_folds):
                end_idx = start_idx + fold_size + (1 if i < remainder else 0)
                folds[i].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        for i in range(self.num_folds):
            train_idx: list[int] = []
            test_idx: list[int] = []
            for j in range(self.num_folds):
                if i == j:
                    test_idx.extend(folds[j])
                else:
                    train_idx.extend(folds[j])
            yield train_idx, test_idx
