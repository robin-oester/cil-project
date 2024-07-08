import pathlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH

from .utils import write_predictions_to_csv


class RatingPredictor(ABC):

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict the rating for the given (user_id, movie_id) pairs.

        :param inputs: numpy array of shape (N, 2) consisting of (user_id, movie_id) pairs.
        :return: (N, 1) numpy array with the predicted ratings.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the predictor.

        :return: predictor name.
        """

        raise NotImplementedError()

    def generate_predictions(self, input_file_name: str, fold_idx: Optional[int] = None) -> pathlib.Path:
        """
        Generate predictions for the given input file and store them.

        :param input_file_name: file name of the submission dataset.
        :param fold_idx: index of the fold. If None, the model's predictions are not part of a k-fold evaluation.
        :return: path to the generated output file.
        """

        input_path = DATA_PATH / input_file_name
        submission_dataset = SubmissionDataset(input_path)
        predictions = self.predict(submission_dataset.inputs)

        assert predictions.shape[0] == submission_dataset.inputs.shape[0]
        assert submission_dataset.predictions.shape == predictions.shape, (
            f"Predictions shape {predictions.shape} does not match submission "
            f"dataset shape {submission_dataset.predictions.shape}."
        )

        submission_dataset.predictions = predictions

        return write_predictions_to_csv(submission_dataset, self.get_name(), fold_idx)
