from typing import Any

import torch
from cil_project.utils import ModelInitializationError, validate_parameter_types

from .abstract_model import AbstractModel


class Autoencoder(AbstractModel):
    """
    Autoencoder model for collaborative filtering.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)
        self.encoder = Encoder(self.num_movies, self.encoding_size)
        self.decoder = Decoder(self.num_movies, self.encoding_size)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("num_movies", int),
                ("encoding_size", int),
            ],
        )

        self.num_movies = hyperparameters["num_movies"]
        self.encoding_size = hyperparameters["encoding_size"]

        if self.num_movies <= 0:
            raise ModelInitializationError("num_movies", "Parameter should be positive")
        if self.encoding_size <= 0:
            raise ModelInitializationError("encoding_size", "Parameter should be positive")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class Encoder(torch.nn.Module):
    """
    Encode the movie ratings for a given user.
    """

    def __init__(self, input_size: int, encoding_size: int = 10) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, encoding_size),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(torch.nn.Module):
    """
    Decode the encoded movie ratings to the original ratings of the user.
    """

    def __init__(self, output_size: int, encoding_size: int = 10) -> None:
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_size, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, output_size),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
