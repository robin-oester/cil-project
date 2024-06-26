from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from cil_project.dataset import RatingsDataset
from cil_project.ensembling import RatingPredictor
from cil_project.neural_filtering.models import AbstractModel
from cil_project.utils import MAX_RATING, MIN_RATING, rmse


class AbstractEvaluator(RatingPredictor):
    """
    Abstract class for evaluating a model on a dataset.
    """

    def __init__(
        self,
        model: AbstractModel,
        batch_size: int,
        dataset: RatingsDataset,
        val_dataset: Optional[RatingsDataset],
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings for the given inputs.

        :param inputs: The inputs for which to predict the ratings (shape: (N, 2)).
        :return: array of shape (N, 1) containing the predicted normalized ratings.
        """

        raise NotImplementedError()

    def evaluate(self) -> float:
        """
        Evaluates the model on the validation dataset.

        :return: score of the model on the validation dataset.
        """

        assert self.val_dataset is not None, "Validation dataset is required for evaluation."

        predictions = self.predict(self.val_dataset.get_inputs())
        return rmse(self.val_dataset.get_targets(), predictions)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the denormalized and clipped ratings for the given inputs.

        :param inputs: The inputs for which to predict the ratings (shape: (N, 2)).
        :return: array of shape (N, 1) containing the predicted normalized ratings.
        """

        assert inputs.shape[1] == 2, "Inputs must have shape (N, 2)."

        self.model.to(self.device)
        predictions = self._predict(inputs)

        denormalized_predictions = self.dataset.denormalize_predictions(inputs, predictions)

        return np.clip(denormalized_predictions, MIN_RATING, MAX_RATING)

    def get_name(self) -> str:
        return self.model.__class__.__name__
