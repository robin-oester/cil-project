from math import sqrt
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from myfm import MyFMRegressor  # pylint: disable=E0401
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


# pylint: disable=too-many-locals
def train(
    dataset: pd.DataFrame,
    iterator: Iterator[Tuple[np.ndarray, np.ndarray]],
    rank: int = 4,
    n_iter: int = 300,
    n_kept_samples: int = 200,
) -> None:
    print("Training Bayesian Factorization Machine...")
    feature_columns = ["user", "movie"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    x = ohe.fit_transform(dataset[feature_columns])
    y = dataset["rating"].values

    rmse_values = []

    for train_indices, test_indices in iterator:
        print(f"Training on {len(train_indices)} samples, testing on {len(test_indices)} samples...")
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        fm = MyFMRegressor(rank=rank, random_seed=42)
        fm.fit(x_train, y_train, n_iter=n_iter, n_kept_samples=n_kept_samples)

        y_pred = fm.predict(x_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)
        print(f"RMSE: {rmse}")

    average_rmse = sum(rmse_values) / len(rmse_values)
    print(f"Average RMSE: {average_rmse}")
