import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

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

    @abstractmethod
    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        """
        Initialize parameters of the model.

        :param hyperparameters: the hyperparameters provided to the model
        """

        raise NotImplementedError()

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass of the model.

        :param inputs: (batched) input to the model.
        :return: the prediction.
        """

        raise NotImplementedError()
