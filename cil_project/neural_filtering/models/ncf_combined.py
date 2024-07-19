from typing import Any

import torch
from cil_project.utils import NUM_MOVIES, NUM_USERS, ModelInitializationError, validate_parameter_types

from .abstract_model import AbstractModel


class NCFCombined(AbstractModel):
    """
    Implements the PyTorch version of the Neural Collaborative Filtering (NCF) model found in
    https://arxiv.org/abs/1708.05031.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        """
        Constructor for the baseline implementation of NCF.
        Requires:
        - `predictive_factor` - the size of the predictive factor (corresponds to the embedding dimension).
        - `alpha` - the value used for blending GMF with MLP.

        :param hyperparameters: hyperparameters for the model.
        """

        super().__init__(hyperparameters)

        # GMF and MLP model
        self.gmf = NCFGMFModel(hyperparameters)
        self.mlp = NCFMLPModel(hyperparameters)

        self.dropout = torch.nn.Dropout(0.5)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("predictive_factor", int),
                ("alpha", float),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]
        self.alpha = hyperparameters["alpha"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")
        if self.alpha < 0 or self.alpha > 1:
            raise ModelInitializationError("alpha", "Parameter should be in range [0, 1]")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # GMF
        x1 = self.gmf(inputs)
        x1 = self.dropout(x1)

        # MLP
        x2 = self.mlp(inputs)
        x2 = self.dropout(x2)

        # NCF
        y = self.alpha * x1 + (1.0 - self.alpha) * x2
        return y


class NCFGMFModel(AbstractModel):
    """
    General matrix factorization layer that computes a generalized dot product of user and movie
    embeddings.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

        self.gmf_user_embedding = torch.nn.Embedding(NUM_USERS, self.predictive_factor)
        self.gmf_movie_embedding = torch.nn.Embedding(NUM_MOVIES, self.predictive_factor)
        self.output_layer = torch.nn.Linear(self.predictive_factor, 1, bias=False)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("predictive_factor", int),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        users = inputs[:, 0]
        movies = inputs[:, 1]

        user_embedding = self.gmf_user_embedding(users)
        movie_embedding = self.gmf_movie_embedding(movies)

        x = torch.mul(user_embedding, movie_embedding)
        x = self.output_layer(x)
        return x


class NCFMLPModel(AbstractModel):
    """
    MLP model to capture non-linear dependencies between user and movie embeddings.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

        self.mlp_user_embedding = torch.nn.Embedding(NUM_USERS, self.predictive_factor // 2)
        self.mlp_movie_embedding = torch.nn.Embedding(NUM_MOVIES, self.predictive_factor // 2)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.predictive_factor, self.predictive_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(self.predictive_factor, self.predictive_factor),
            torch.nn.ReLU(),
        )

        self.output_layer = torch.nn.Linear(self.predictive_factor, 1, bias=False)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("predictive_factor", int),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        users = inputs[:, 0]
        movies = inputs[:, 1]
        user_embedding = self.mlp_user_embedding(users)
        movie_embedding = self.mlp_movie_embedding(movies)

        x = torch.cat([user_embedding, movie_embedding], dim=1)
        x = self.fc(x)

        x = self.output_layer(x)
        return x
