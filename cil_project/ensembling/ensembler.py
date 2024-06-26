import logging
import pathlib

import numpy as np
from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH

from .utils import write_predictions_to_csv

logger = logging.getLogger(__name__)


class Ensembler:
    """
    Class to combine predictions from different predictors.
    """

    @staticmethod
    def combine_predictions(predictor_names: list[str], input_file_name: str) -> pathlib.Path:
        """
        Generate predictions for the given input file and write them to the output folder.
        Takes the most recent prediction file for each predictor.

        :param predictor_names: names of the predictors.
        :param input_file_name: file name of the submission dataset.
        :return: path to the generated output file.
        """

        submission_dataset = SubmissionDataset(DATA_PATH / input_file_name)

        # perform predictions for all predictors
        for predictor_name in predictor_names:
            predictor_file = Ensembler.find_newest_predictor_file(predictor_name)

            predicted_dataset = SubmissionDataset(predictor_file, set_values_to_zero=False)

            assert np.array_equal(
                submission_dataset.inputs, predicted_dataset.inputs
            ), "Inputs for predictions do not match."
            assert len(submission_dataset.predictions) == len(
                predicted_dataset.predictions
            ), "Prediction lengths do not match."

            submission_dataset.predictions += predicted_dataset.predictions

        submission_dataset.predictions /= len(predictor_names)
        return write_predictions_to_csv(submission_dataset, "Ensembler")

    @staticmethod
    def find_newest_predictor_file(predictor_name: str) -> pathlib.Path:
        """
        Find the newest predictor file for the given predictor name.

        :param predictor_name: name of the predictor.
        :return: path to the newest predictor file, never None.
        """

        # Pattern to match "predictorName_TIMESTAMP.csv"
        pattern = f"{predictor_name}_*.csv"

        # List all matching files
        files = list(DATA_PATH.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found in {DATA_PATH} matching pattern '{pattern}'.")

        # Find the file with the largest timestamp
        newest_file = max(files, key=Ensembler.extract_timestamp, default=None)
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
