import pathlib
from abc import ABC, abstractmethod

import numpy as np
from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH


class AbstractEnsembler(ABC):

    def __init__(self, model_names: list[str]):
        self.model_names = model_names

    @abstractmethod
    def predict(self, submission_dataset: SubmissionDataset) -> pathlib.Path:
        """
        Predict the submission dataset.

        :param submission_dataset: the submission dataset to predict.
        """

        raise NotImplementedError()

    @staticmethod
    def load_predictions(paths: list[pathlib.Path], inputs: np.ndarray) -> np.ndarray:
        """
        Load the predictions from the specified paths and check, whether they contain the correct inputs for prediction.
        Attention! Need to have at least one entry in inputs (usually fulfilled).

        :param paths: M file paths containing predictions.
        :param inputs: an (N, 2) array containing the inputs for N predictions.
        :return: (N, M) array consisting of the N predictions for the M files.
        """

        assert len(paths) > 0, "Must specify at least one file with predictions."

        predictions = []
        for path in paths:
            assert path.is_file(), f"No file found at {path}."

            submission = SubmissionDataset(path, set_values_to_zero=False)

            assert np.array_equal(inputs, submission.inputs), "Inputs for predictions do not match."
            assert inputs.shape[0] == submission.predictions.shape[0], "Prediction lengths do not match."

            predictions.append(submission.predictions)

        return np.stack(predictions, axis=0).squeeze().transpose()

    @staticmethod
    def find_newest_predictor_file(model_name: str) -> pathlib.Path:
        """
        Find the newest prediction file for the given predictor name.
        Checks for files of the form "<model_name>_TIMESTAMP.csv".

        :param model_name: name of the model.
        :return: path to the newest prediction file otherwise raises ValueError.
        """

        # Pattern to match "predictorName_TIMESTAMP.csv"
        pattern = f"{model_name}_*.csv"

        # List all matching files
        files = list(DATA_PATH.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found in {DATA_PATH} matching pattern '{pattern}'.")

        # Find the file with the largest timestamp
        newest_file = max(files, key=AbstractEnsembler.extract_timestamp, default=None)
        if newest_file is None:
            raise ValueError(f"No valid file found in {DATA_PATH} matching pattern '{pattern}'.")

        # Return the newest file
        return newest_file

    @staticmethod
    def extract_timestamp(file_path: pathlib.Path) -> int:
        # Extract the part of the filename between the last underscore and ".csv"
        filename = file_path.name
        start = filename.rfind("_") + 1
        end = filename.rfind(".csv")
        timestamp = filename[start:end]
        return int(timestamp) if timestamp.isdigit() else 0
