# This code is adapted from https://myfm.readthedocs.io/en/stable/index.html

import os
import pickle
from collections import defaultdict
from math import sqrt
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from dataset import RatingsDataset
from myfm import MyFMOrderedProbit, MyFMRegressor, RelationBlock  # pylint: disable=E0401
from scipy import sparse as sps
from sklearn.metrics import mean_squared_error


class BayesianFactorizationMachine:
    # pylint: disable=too-many-locals
    def __init__(
        self, dataset: RatingsDataset, iterator: Iterator[Tuple[np.ndarray, np.ndarray]], model_name: str = "bfm_model"
    ) -> None:
        # super().__init__(self.__class__.__name__)
        self.model_name = model_name
        self.model = self.load_model()
        self.data = dataset.get_data_frame()
        self.iterator = iterator
        self.ohe = dataset.get_one_hot_encoder()

    def load_model(self) -> MyFMRegressor | None:
        model_path = os.path.join(os.path.dirname(__file__), self.model_name + ".pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        return None

    # pylint: disable=R0912
    # pylint: disable=R0915
    def train(
        self,
        rank: int = 4,
        n_iter: int = 300,
        n_kept_samples: int = 200,
        grouped: bool = False,
        implicit: bool = False,
        ordered_probit: bool = False,
        save_model: bool = True,
        output_name: str = "bfm_model",
    ) -> None:
        print("Training Bayesian Factorization Machine...")
        feature_columns = ["user", "movie"]

        x = self.ohe.transform(self.data[feature_columns])
        y = self.data["rating"].values

        rmse_values = []

        for train_indices, test_indices in self.iterator:
            print(f"Training on {len(train_indices)} samples, testing on {len(test_indices)} samples...")

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Define the model
            if ordered_probit:
                output_name = output_name + "_ordered_probit"
                fm = MyFMOrderedProbit(rank=rank, random_seed=42)
                y_train = y_train - 1
            else:
                output_name = output_name + "_normal"
                fm = MyFMRegressor(rank=rank, random_seed=42)

            if implicit:
                print("Implicit")
                output_name = output_name + "_implicit"
                df_train = self.data.iloc[train_indices]
                df_test = self.data.iloc[test_indices]

                # index "0" is reserved for unknown ids.
                user_to_index = defaultdict(
                    lambda: 0, {uid: i + 1 for i, uid in enumerate(np.unique(df_train["user"]))}
                )
                movie_to_index = defaultdict(
                    lambda: 0, {mid: i + 1 for i, mid in enumerate(np.unique(df_train["movie"]))}
                )
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

                if grouped:
                    print("Grouped")
                    output_name = output_name + "_grouped"
                    group_shapes = [user_id_size, movie_id_size, user_id_size, movie_id_size]
                    fm.fit(
                        None,
                        y_train,
                        X_rel=[block_user_train, block_movie_train],
                        n_iter=n_iter,
                        n_kept_samples=n_kept_samples,
                        group_shapes=group_shapes,
                    )
                else:
                    print("Ungrouped")
                    output_name = output_name + "_ungrouped"
                    fm.fit(
                        None,
                        y_train,
                        X_rel=[block_user_train, block_movie_train],
                        n_iter=n_iter,
                        n_kept_samples=n_kept_samples,
                    )

                # Make predictions
                if ordered_probit:
                    print("Predicting ordered probit")
                    p_ordinal = fm.predict_proba(None, X_rel=[block_user_test, block_movie_test])
                    y_pred = p_ordinal.dot(np.arange(1, 6))
                else:
                    print("Predicting normal")
                    y_pred = fm.predict(None, X_rel=[block_user_test, block_movie_test])

            else:
                print("Standard")
                if grouped:
                    print("Grouped")
                    output_name = output_name + "_grouped"
                    group_shapes = [len(group) for group in self.ohe.categories_]
                    fm.fit(x_train, y_train, n_iter=n_iter, n_kept_samples=n_kept_samples, group_shapes=group_shapes)
                else:
                    print("Ungrouped")
                    output_name = output_name + "_ungrouped"
                    fm.fit(x_train, y_train, n_iter=n_iter, n_kept_samples=n_kept_samples)

                # Make predictions
                if ordered_probit:
                    print("Predicting ordered probit")
                    p_ordinal = fm.predict_proba(x_test)
                    y_pred = p_ordinal.dot(np.arange(1, 6))
                else:
                    print("Predicting normal")
                    y_pred = fm.predict(x_test)
                y_pred = fm.predict(x_test)

            rmse = sqrt(mean_squared_error(y_test, y_pred))
            rmse_values.append(rmse)
            print(f"RMSE: {rmse}")

            if save_model:
                # Save the fitted model
                model_path = os.path.join(os.path.dirname(__file__), output_name + ".pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(fm, f)

            self.model = fm

        average_rmse = sum(rmse_values) / len(rmse_values)
        print(f"Average RMSE: {average_rmse}")

    def predict(self, x: tuple[int, int]) -> float:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        x_df = pd.DataFrame([x], columns=["user", "movie"])

        x_transformed = self.ohe.transform(x_df)
        return self.model.predict(x_transformed)[0]

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

        x_ii = sps.lil_matrix((len(movie_ids), user_id_size))
        for index, movie_id in enumerate(movie_ids):
            watched_users = movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                x_ii[index, user_to_index[uid]] = normalizer
        xs.append(x_ii)

        return sps.hstack(xs, format="csr")
