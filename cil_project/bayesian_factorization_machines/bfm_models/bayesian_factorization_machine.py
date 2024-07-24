import logging

import numpy as np
from cil_project.dataset import RatingsDataset
from cil_project.utils import MAX_RATING, MIN_RATING, rmse
from myfm import MyFMRegressor

from .abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class BayesianFactorizationMachine(AbstractModel):
    def __init__(
        self,
        rank: int = 4,
        num_bins: int = 50,
        num_clusters: int = 5,
        grouped: bool = False,
        implicit: bool = False,
        statistical_features: bool = False,
        kmeans: bool = False,
    ) -> None:
        super().__init__(rank, num_bins, num_clusters, grouped, implicit, statistical_features, kmeans)

        self.model = MyFMRegressor(rank=rank, random_seed=42)
        self.train_dataset = None

    # pylint: disable=R0914
    def train(
        self,
        train_dataset: RatingsDataset,
        test_dataset: RatingsDataset,
        n_iter: int = 300,
    ) -> float:
        logger.info("Training Bayesian Factorization Machine...")

        self.train_dataset = train_dataset
        y_train = train_dataset.get_targets().reshape(1, -1)[0]
        y_test = test_dataset.get_targets().reshape(1, -1)[0]

        # Get the one-hot encoding of the features
        x_rel_train = self.get_features(train_dataset, train_dataset)
        x_rel_test = self.get_features(test_dataset, train_dataset)
        group_shapes = self.get_group_shapes()

        n_kept_samples = n_iter

        self.model.fit(
            None,
            y_train,
            X_rel=x_rel_train,
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            group_shapes=group_shapes,
        )

        y_pred = self.model.predict(None, X_rel=x_rel_test, n_workers=8)
        y_pred = np.clip(y_pred, MIN_RATING, MAX_RATING)

        error = rmse(y_test, y_pred)
        logger.info(f"RMSE: {error}")
        return error

    def final_train(
        self,
        dataset: RatingsDataset,
        n_iter: int = 300,
    ) -> None:
        logger.info("Training Bayesian Factorization Machine...")

        self.train_dataset = dataset
        y_train = dataset.get_targets().reshape(1, -1)[0]
        x_rel_train = self.get_features(dataset, dataset)
        group_shapes = self.get_group_shapes()
        n_kept_samples = n_iter

        self.model.fit(
            None,
            y_train,
            X_rel=x_rel_train,
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            group_shapes=group_shapes,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.train_dataset is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        pred_dataset = RatingsDataset(inputs, np.zeros((inputs.shape[0], 1)))
        x_rel = self.get_features(pred_dataset, self.train_dataset)
        y_pred = self.model.predict(None, X_rel=x_rel, n_workers=8)
        return np.clip(y_pred, MIN_RATING, MAX_RATING).reshape(-1, 1)

    def get_name(self) -> str:
        return self.model.__class__.__name__
