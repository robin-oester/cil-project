import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
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
    ) -> None:
        """
        Initializes a new bfm given some model configuration options.

        :param hyperparameters: consists of all model configuration options.
        """

        super().__init__()
        self.rank = rank
        self.grouped = grouped
        self.implicit = implicit

    @abstractmethod
    def train(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_iter: int = 300,
    ) -> None:
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
        user_ids: list[int],
        user_to_index: dict[int, int],
        user_vs_watched: dict[int, list[int]],
        movie_to_index: dict[int, int],
        user_id_size: int,
        movie_id_size: int,
    ) -> sps.csr_matrix:
        xs = []
        x_uid = sps.lil_matrix((len(user_ids), user_id_size))
        for index, user_id in enumerate(user_ids):
            x_uid[index, user_to_index[user_id]] = 1
        xs.append(x_uid)

        if self.implicit:
            x_iu = sps.lil_matrix((len(user_ids), movie_id_size))
            for index, user_id in enumerate(user_ids):
                watched_movies = user_vs_watched.get(user_id, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for uid in watched_movies:
                    x_iu[index, movie_to_index[uid]] = normalizer
            xs.append(x_iu)

        return sps.hstack(xs, format="csr")

    def augment_movie_id(
        self,
        movie_ids: list[int],
        user_to_index: dict[int, int],
        movie_vs_watched: dict[int, list[int]],
        movie_to_index: dict[int, int],
        user_id_size: int,
        movie_id_size: int,
    ) -> sps.csr_matrix:
        xs = []
        x_movie = sps.lil_matrix((len(movie_ids), movie_id_size))
        for index, movie_id in enumerate(movie_ids):
            x_movie[index, movie_to_index[movie_id]] = 1
        xs.append(x_movie)

        if self.implicit:
            x_ii = sps.lil_matrix((len(movie_ids), user_id_size))
            for index, movie_id in enumerate(movie_ids):
                watched_users = movie_vs_watched.get(movie_id, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    x_ii[index, user_to_index[uid]] = normalizer
            xs.append(x_ii)

        return sps.hstack(xs, format="csr")

    # pylint: disable=R0914
    def get_features(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[RelationBlock, RelationBlock, RelationBlock, RelationBlock, list[int]]:
        # index "0" is reserved for unknown ids.
        user_to_index = defaultdict(lambda: 0, {uid: i + 1 for i, uid in enumerate(np.unique(df_train["user"]))})
        movie_to_index = defaultdict(lambda: 0, {mid: i + 1 for i, mid in enumerate(np.unique(df_train["movie"]))})
        user_id_size = len(user_to_index) + 1
        movie_id_size = len(movie_to_index) + 1
        movie_vs_watched: dict[int, list[int]] = {}
        user_vs_watched: dict[int, list[int]] = {}
        for row in df_train.itertuples():
            user_id = row.user
            movie_id = row.movie
            movie_vs_watched.setdefault(movie_id, []).append(user_id)
            user_vs_watched.setdefault(user_id, []).append(movie_id)

        train_uid_unique, train_uid_index = np.unique(df_train["user"], return_inverse=True)
        train_mid_unique, train_mid_index = np.unique(df_train["movie"], return_inverse=True)
        user_data_train = self.augment_user_id(
            train_uid_unique, user_to_index, user_vs_watched, movie_to_index, user_id_size, movie_id_size
        )
        movie_data_train = self.augment_movie_id(
            train_mid_unique, user_to_index, movie_vs_watched, movie_to_index, user_id_size, movie_id_size
        )

        test_uid_unique, test_uid_index = np.unique(df_test["user"], return_inverse=True)
        test_mid_unique, test_mid_index = np.unique(df_test["movie"], return_inverse=True)
        user_data_test = self.augment_user_id(
            test_uid_unique, user_to_index, user_vs_watched, movie_to_index, user_id_size, movie_id_size
        )
        movie_data_test = self.augment_movie_id(
            test_mid_unique, user_to_index, movie_vs_watched, movie_to_index, user_id_size, movie_id_size
        )

        block_user_train = RelationBlock(train_uid_index, user_data_train)
        block_movie_train = RelationBlock(train_mid_index, movie_data_train)
        block_user_test = RelationBlock(test_uid_index, user_data_test)
        block_movie_test = RelationBlock(test_mid_index, movie_data_test)

        if self.grouped:
            if self.implicit:
                group_shapes = [user_id_size, movie_id_size, user_id_size, movie_id_size]
            else:
                group_shapes = [user_id_size, movie_id_size]
        else:
            group_shapes = None
        return block_user_train, block_movie_train, block_user_test, block_movie_test, group_shapes
