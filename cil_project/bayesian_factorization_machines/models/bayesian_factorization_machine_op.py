# This code is adapted from https://myfm.readthedocs.io/en/stable/index.html

import numpy as np
from cil_project.dataset import RatingsDataset, SubmissionDataset
from cil_project.ensembling import RatingPredictor
from cil_project.utils import DATA_PATH, SUBMISSION_FILE_NAME, rmse
from myfm import MyFMOrderedProbit  # pylint: disable=E0401

from .abstract_model import AbstractModel


class BayesianFactorizationMachineOP(AbstractModel, RatingPredictor):
    def __init__(
        self,
        rank: int = 4,
        grouped: bool = False,
        implicit: bool = False,
        statistical_features: bool = False,
    ) -> None:
        super().__init__(rank, grouped, implicit, statistical_features)

        self.model = MyFMOrderedProbit(rank=rank, random_seed=42)
        self.train_dataset = None

    # pylint: disable=R0914
    def train(
        self,
        train_dataset: RatingsDataset,
        test_dataset: RatingsDataset,
        n_iter: int = 300,
    ) -> float:
        print("Training Bayesian Factorization Machine...")

        y_train = train_dataset.get_targets().reshape(1, -1)[0]
        # Since the model expects the ratings to be in the range [1, 5], we need to scale them to [0, 4]
        y_train = y_train - 1
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

        p_ordinal = self.model.predict_proba(None, X_rel=x_rel_test)
        y_pred = p_ordinal.dot(np.arange(1, 6))

        error = rmse(y_test, y_pred)
        print(f"RMSE: {error}")
        return error

    def final_train(
        self,
        dataset: RatingsDataset,
        n_iter: int = 300,
    ) -> None:
        print("Training Bayesian Factorization Machine...")

        self.train_dataset = dataset
        y_train = dataset.get_targets().reshape(1, -1)[0]
        y_train = y_train - 1
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Only use this method for predictions using the whole dataset
        if self.train_dataset is None:
            raise ValueError("Model is not trained yet. Call the 'train' method to train the model.")

        input_path = DATA_PATH / SUBMISSION_FILE_NAME
        submission_dataset = SubmissionDataset(input_path)
        x_rel = self.get_features(submission_dataset, self.train_dataset)
        p_ordinal = self.model.predict_proba(None, X_rel=x_rel)
        y_pred = p_ordinal.dot(np.arange(1, 6))
        return y_pred.reshape(-1, 1)

    def get_name(self) -> str:
        return self.model.__class__.__name__
