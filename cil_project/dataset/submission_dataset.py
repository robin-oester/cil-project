import csv
import logging
import pathlib
import re

import numpy as np

logger = logging.getLogger(__name__)

REGEX_PATTERN = r"r(\d+)_c(\d+)"


class SubmissionDataset:
    """
    Dataset holding the submission tuples (and possibly also predictions).
    """

    def __init__(self, inputs: np.ndarray, predictions: np.ndarray):
        """
        Initialize the submission dataset.

        :param inputs: the inputs of the submission having shape (N, 2) for (user_id, movie_id)-pairs.
        :param predictions: the associated predictions (shape (N, 1)).
        """

        self.inputs = inputs  # shape: (N, 2)
        self.predictions = predictions  # shape: (N, 1)

    @classmethod
    def from_file(cls, file_path: pathlib.Path, set_values_to_zero: bool = True) -> "SubmissionDataset":
        """
        Load the submission dataset from the given file path.

        :param file_path: path to the submission file.
        :param set_values_to_zero: if the predictions should not be loaded.
        :return: the submission dataset.
        """

        inputs: list[np.ndarray] = []
        predictions: list[float] = []

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                id_str, rating_str = row
                match = re.match(REGEX_PATTERN, id_str)
                if match:
                    # both are 1-based
                    user_idx = int(match.group(1)) - 1
                    movie_idx = int(match.group(2)) - 1
                else:
                    raise ValueError(f"Id '{id_str}' does not match the expected pattern.")
                rating = 0.0 if set_values_to_zero else float(rating_str.strip())

                inputs.append(np.array([user_idx, movie_idx]))
                predictions.append(rating)

        logging.info(f"Loaded a total of {len(predictions)} entries.")
        return cls(np.array(inputs), np.array(predictions, dtype=np.float32).reshape((-1, 1)))

    def __len__(self) -> int:
        return len(self.predictions)
