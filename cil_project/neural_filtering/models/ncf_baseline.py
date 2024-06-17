from typing import Any

import torch

from .abstract_model import AbstractModel
from .model_initialization_error import ModelInitializationError


class NCFBaseline(AbstractModel):
    """
    Implements the PyTorch version of the Neural Collaborative Filtering (NCF) model found in
    https://arxiv.org/abs/1708.05031. Uses tanh activation to get normalized rating values in [-1, 1].
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

        # GMF and MLP model
        self.gmf = GMFLayer(self.num_users, self.num_movies, self.predictive_factor)
        self.mlp = MLPLayer(self.num_users, self.num_movies, self.predictive_factor)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        AbstractModel.validate_parameter_types(
            hyperparameters,
            [
                ("num_users", int),
                ("num_movies", int),
                ("predictive_factor", int),
                ("alpha", float),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]
        self.num_users = hyperparameters["num_users"]
        self.num_movies = hyperparameters["num_movies"]
        self.alpha = hyperparameters["alpha"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")
        if self.num_users <= 0:
            raise ModelInitializationError("num_users", "Parameter should be positive")
        if self.num_movies <= 0:
            raise ModelInitializationError("num_movies", "Parameter should be positive")
        if self.alpha < 0 or self.alpha > 1:
            raise ModelInitializationError("alpha", "Parameter should be in range [0, 1]")

    def predict(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        # GMF
        x1 = self.gmf(users, movies)

        # MLP
        x2 = self.mlp(users, movies)

        # NCF
        logit = self.alpha * x1 + (1.0 - self.alpha) * x2
        return torch.tanh(logit)


class NCFGMFModel(AbstractModel):
    """
    Generalized matrix factorization model.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

        self.gmf = GMFLayer(self.num_users, self.num_movies, self.predictive_factor)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        AbstractModel.validate_parameter_types(
            hyperparameters,
            [
                ("num_users", int),
                ("num_movies", int),
                ("predictive_factor", int),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]
        self.num_users = hyperparameters["num_users"]
        self.num_movies = hyperparameters["num_movies"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")
        if self.num_users <= 0:
            raise ModelInitializationError("num_users", "Parameter should be positive")
        if self.num_movies <= 0:
            raise ModelInitializationError("num_movies", "Parameter should be positive")

    def predict(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        x = self.gmf(users, movies)
        x = torch.tanh(x)

        return x


class NCFMLPModel(AbstractModel):
    """
    MLP model to capture non-linear dependencies between user and movie embeddings.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

        self.mlp = MLPLayer(self.num_users, self.num_movies, self.predictive_factor)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        AbstractModel.validate_parameter_types(
            hyperparameters,
            [
                ("num_users", int),
                ("num_movies", int),
                ("predictive_factor", int),
            ],
        )

        self.predictive_factor = hyperparameters["predictive_factor"]
        self.num_users = hyperparameters["num_users"]
        self.num_movies = hyperparameters["num_movies"]

        if self.predictive_factor <= 0:
            raise ModelInitializationError("predictive_factor", "Parameter should be positive")
        if self.num_users <= 0:
            raise ModelInitializationError("num_users", "Parameter should be positive")
        if self.num_movies <= 0:
            raise ModelInitializationError("num_movies", "Parameter should be positive")

    def predict(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        x = self.mlp(users, movies)
        x = torch.tanh(x)

        return x


class GMFLayer(torch.nn.Module):
    """
    General matrix factorization layer that computes a generalized dot product of user and movie
    embeddings.
    """

    def __init__(self, num_users: int, num_movies: int, predictive_factor: int) -> None:
        super().__init__()

        self.gmf_user_embedding = torch.nn.Embedding(num_users, predictive_factor)
        self.gmf_movie_embedding = torch.nn.Embedding(num_movies, predictive_factor)
        self.output_layer = torch.nn.Linear(predictive_factor, 1, bias=False)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        user_embedding = self.gmf_user_embedding(users)
        movie_embedding = self.gmf_movie_embedding(movies)

        x = torch.mul(user_embedding, movie_embedding)
        x = self.output_layer(x)
        return x


class MLPLayer(torch.nn.Module):
    """
    MLP layer of the NCF model. Captures non-linear dependencies between user and movie embeddings.
    It uses 3 hierarchical fully-connected layers with ReLU activations.
    """

    def __init__(self, num_users: int, num_movies: int, predictive_factor: int) -> None:
        super().__init__()

        layers = [predictive_factor * 4, predictive_factor * 2, predictive_factor]

        self.mlp_user_embedding = torch.nn.Embedding(num_users, predictive_factor * 2)
        self.mlp_movie_embedding = torch.nn.Embedding(num_movies, predictive_factor * 2)

        self.fc_layers = torch.nn.ModuleList()
        for input_shape, output_shape in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(torch.nn.Linear(input_shape, output_shape))
        self.output_layer = torch.nn.Linear(predictive_factor, 1, bias=False)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        user_embedding = self.mlp_user_embedding(users)
        movie_embedding = self.mlp_movie_embedding(movies)

        x = torch.cat([user_embedding, movie_embedding], dim=1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x
