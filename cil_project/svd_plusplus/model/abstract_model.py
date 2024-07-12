import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent.parent / "trainer" / "checkpoints"


class AbstractModel(ABC, torch.nn.Module):
    """
    Abstract model class for SVDPP.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        """
        Initializes a new neural network given some model configuration options.

        :param hyperparameters: consists of all model configuration options.
        """

        super().__init__()
        self.hyperparameters = hyperparameters
        self._initialize_parameters(hyperparameters)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_name: str) -> "AbstractModel":
        """
        Load a model from a checkpoint generated from a trainer.

        :param checkpoint_name: the file name of the checkpoint.
        :return: the loaded model.
        """

        loaded_dict: dict = torch.load(CHECKPOINT_PATH / checkpoint_name)
        hyperparameters = loaded_dict["hyperparameters"]
        model = cls(hyperparameters)
        model.load_state_dict(loaded_dict["model"])

        return model

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
