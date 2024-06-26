import pathlib
from typing import Any

import numpy as np
import torch

from .model_initialization_error import ModelInitializationError

# Constants
CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent / ".." / "neural_filtering" / "checkpoints"
DATA_PATH = pathlib.Path(__file__).resolve().parent / ".." / ".." / "data"
FULL_SERIALIZED_DATASET_NAME = "serialized_ratings"
SUBMISSION_FILE_NAME = "sampleSubmission.csv"

# dataset constants
NUM_USERS = 10_000
NUM_MOVIES = 1_000
MIN_RATING = 1.0
MAX_RATING = 5.0


def masked_mse(y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean squared error between the true values and the predicted values, only considering the masked
    values.

    :param y: the true values.
    :param y_hat: the predicted values.
    :param mask: the mask indicating which values to consider.
    :return: the masked mean squared error.
    """
    return torch.sum(mask * (y - y_hat) ** 2) / torch.sum(mask)


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the root mean squared error between the true values and the predicted values.

    :param y: the true values.
    :param y_hat: the predicted values.
    :return: the root mean squared error.
    """
    return np.sqrt(np.mean((y - y_hat) ** 2))


def masked_rmse(y: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the root mean squared error between the true values and the predicted values, only considering the
    masked values.

    :param y: the true values.
    :param y_hat: the predicted values.
    :param mask: the mask indicating which values to consider.
    :return: the masked root mean squared error.
    """
    return np.sqrt(np.sum((mask * (y - y_hat)) ** 2) / float(np.sum(mask)))


def validate_parameter_types(hyperparameters: dict[str, Any], types: list[tuple[str, type]]) -> None:
    """
    Helper function to validate hyperparameters. I.e., that they present and have the correct type.

    :param hyperparameters: the hyperparameters of the model.
    :param types: consists of the parameter names that must be available and their associated type.
    """

    for param_name, param_type in types:
        if param_name not in hyperparameters:
            raise ModelInitializationError(param_name, "Parameter not found")
        if not isinstance(hyperparameters[param_name], param_type):
            raise ModelInitializationError(param_name, "Parameter doesn't match with expected type")
