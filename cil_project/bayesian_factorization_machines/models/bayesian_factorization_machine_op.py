# This code is adapted from https://myfm.readthedocs.io/en/stable/index.html

import numpy as np
import pandas as pd
from cil_project.dataset import RatingsDataset
from cil_project.ensembling import RatingPredictor
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, rmse
from myfm import MyFMOrderedProbit  # pylint: disable=E0401

from .abstract_model import AbstractModel


class BayesianFactorizationMachineOP(AbstractModel, RatingPredictor):
    def __init__(
        self,
        rank: int = 4,
        grouped: bool = False,
        implicit: bool = False,
    ) -> None:
        super().__init__(rank, grouped, implicit)

        self.model = MyFMOrderedProbit(rank=rank, random_seed=42)

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

        # Since the model expects the ratings to be in the range [1, 5], we need to scale them to [0, 4]
        y_train = y_train - 1

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

        p_ordinal = self.model.predict_proba(None, X_rel=[block_user_test, block_movie_test])
        y_pred = p_ordinal.dot(np.arange(1, 6))

        error = rmse(y_test, y_pred)
        print(f"RMSE: {error}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Only use this method for predictions using the whole dataset
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        x_df = pd.DataFrame([x], columns=["user", "movie"])
        dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

        _, _, block_user, block_movie, _ = self.get_features(dataset.get_data_frame()[["user", "movie"]], x_df)
        p_ordinal = self.model.predict_proba(None, X_rel=[block_user, block_movie])
        y_pred = p_ordinal.dot(np.arange(1, 6))
        return y_pred

    def get_name(self) -> str:
        return self.model.__class__.__name__
