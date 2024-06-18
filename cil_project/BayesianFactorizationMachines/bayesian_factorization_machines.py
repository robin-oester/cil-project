import os
import pickle
from math import sqrt
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from dataset import RatingsDataset
from myfm import MyFMRegressor  # pylint: disable=E0401
from sklearn.metrics import mean_squared_error


class BayesianFactorizationMachine:
    # pylint: disable=too-many-locals
    def __init__(self, dataset: RatingsDataset, iterator: Iterator[Tuple[np.ndarray, np.ndarray]]) -> None:
        # super().__init__(self.__class__.__name__)
        self.model = self.load_model()
        self.data = dataset.get_data_frame()
        self.iterator = iterator
        self.ohe = dataset.get_one_hot_encoder()

    def load_model(self) -> MyFMRegressor | None:
        model_path = os.path.join(os.path.dirname(__file__), "bfm_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        return None

    def train(
        self,
        rank: int = 4,
        n_iter: int = 300,
        n_kept_samples: int = 200,
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

            fm = MyFMRegressor(rank=rank, random_seed=42)
            fm.fit(x_train, y_train, n_iter=n_iter, n_kept_samples=n_kept_samples)

            # Save the fitted model
            model_path = os.path.join(os.path.dirname(__file__), "bfm_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(fm, f)
            self.model = fm

            y_pred = fm.predict(x_test)
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            rmse_values.append(rmse)
            print(f"RMSE: {rmse}")

        average_rmse = sum(rmse_values) / len(rmse_values)
        print(f"Average RMSE: {average_rmse}")

    def predict(self, x: tuple[int, int]) -> float:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        x_df = pd.DataFrame([x], columns=["user", "movie"])

        x_transformed = self.ohe.transform(x_df)
        return self.model.predict(x_transformed)[0]
