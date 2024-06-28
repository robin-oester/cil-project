# This code is adapted from https://myfm.readthedocs.io/en/stable/index.html

import numpy as np
import pandas as pd
from cil_project.dataset import RatingsDataset
from cil_project.ensembling import RatingPredictor
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, MAX_RATING, MIN_RATING, rmse
from myfm import MyFMRegressor  # pylint: disable=E0401

from .abstract_model import AbstractModel


class BayesianFactorizationMachine(AbstractModel, RatingPredictor):
    def __init__(
        self,
        rank: int = 4,
        grouped: bool = False,
        implicit: bool = False,
    ) -> None:
        super().__init__(rank, grouped, implicit)

        self.model = MyFMRegressor(rank=rank, random_seed=42)

    # pylint: disable=R0914

    def train(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_iter: int = 300,
    ) -> None:
        print("Training Bayesian Factorization Machine...")

        feature_columns = ["user", "movie"]
        x_train, x_test = df_train[feature_columns], df_test[feature_columns]
        y_train, y_test = df_train["rating"], df_test["rating"]

        print(f"Training on {len(x_train)} samples, testing on {len(x_test)} samples...")

        # Get the one-hot encoding of the features and (optionally) implicit features (as RelationBlocks)

        block_user_train, block_movie_train, block_user_test, block_movie_test, group_shapes = self.get_features(
            df_train, df_test
        )

        n_kept_samples = n_iter

        self.model.fit(
            None,
            y_train,
            X_rel=[block_user_train, block_movie_train],
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            group_shapes=group_shapes,
        )

        y_pred = self.model.predict(None, X_rel=[block_user_test, block_movie_test])

        error = rmse(y_test, y_pred)
        print(f"RMSE: {error}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        x_df = pd.DataFrame([x], columns=["user", "movie"])
        dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

        _, _, block_user, block_movie, _ = self.get_features(dataset.get_data_frame()[["user", "movie"]], x_df)
        y_pred = self.model.predict(None, X_rel=[block_user, block_movie])
        return np.clip(y_pred, MIN_RATING, MAX_RATING)

    def get_name(self) -> str:
        return self.model.__class__.__name__
