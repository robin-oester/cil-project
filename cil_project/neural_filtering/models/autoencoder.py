from typing import Any

import torch
from cil_project.utils import NUM_MOVIES, ModelInitializationError, validate_parameter_types

from .abstract_model import AbstractModel


class Autoencoder(AbstractModel):
    """
    Autoencoder model for collaborative filtering.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        """
        Constructor for the autoencoder.
        Requires:
        - `encoding_size` - the latent space dimension.
        - `p_dropout` - the probability of a neuron set to 0 during training.

        :param hyperparameters: hyperparameters for the model.
        """

        super().__init__(hyperparameters)
        self.encoder = Encoder(self.p_dropout, self.encoding_size)
        self.decoder = Decoder(self.p_dropout, self.encoding_size)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("encoding_size", int),
                ("p_dropout", float),
            ],
        )

        self.encoding_size = hyperparameters["encoding_size"]
        self.p_dropout = hyperparameters["p_dropout"]

        if self.encoding_size <= 0:
            raise ModelInitializationError("encoding_size", "Parameter should be positive")
        if self.p_dropout <= 0 or self.p_dropout >= 1:
            raise ModelInitializationError("p_dropout", "Parameter should be in (0, 1) range")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class Encoder(torch.nn.Module):
    """
    Encode the movie ratings for a given user.
    """

    def __init__(self, p_dropout: float, encoding_size: int = 32) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(NUM_MOVIES, 256),
            torch.nn.Dropout(p_dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, encoding_size),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(torch.nn.Module):
    """
    Decode the encoded movie ratings to the original ratings of the user.
    """

    def __init__(self, p_dropout: float, encoding_size: int = 32) -> None:
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_size, 256),
            torch.nn.Dropout(p_dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, NUM_MOVIES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
