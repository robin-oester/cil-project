import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from .model_initialization_error import ModelInitializationError

logger = logging.getLogger(__name__)


class AbstractModel(ABC, torch.nn.Module):
    """
    Abstract model class for each neural network.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        """
        Initializes a new neural network given some model configuration options.

        :param hyperparameters: consists of all model configuration options.
        """

        super().__init__()
        self.hyperparameters = hyperparameters
        self._initialize_parameters(hyperparameters)

    @staticmethod
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

    @abstractmethod
    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        """
        Initialize parameters of the model.

        :param hyperparameters: the hyperparameters provided to the model
        """

        raise NotImplementedError()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass of the model. For a batch of size B.

        :param inputs: B x 2 tensor for the user and movie indices.
        :return: the prediction.
        """

        return self.predict(inputs[:, 0], inputs[:, 1])

    @abstractmethod
    def predict(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Perform the model prediction for given users and movies.

        :param users: indices of the users in the batch.
        :param movies: indices of the movies in the batch.
        :return: the prediction as a np.float32 tensor.
        """

        raise NotImplementedError()
