from typing import Any

import torch
from cil_project.utils import NUM_MOVIES, NUM_USERS, ModelInitializationError, validate_parameter_types

from .abstract_model import AbstractModel
from .kan import KAN


class KANNCF(AbstractModel):
    """
    KAN-based Neural Collaborative Filtering model.
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        """
        Constructor for KAN-based NCF.
        Requires:
        - `embedding_dim` - the dimension of the embeddings for the user and movie.
        - `hidden_dim` - the dimension of the hidden layers of the KAN layers.

        :param hyperparameters: hyperparameters for the model.
        """

        super().__init__(hyperparameters)

        self.embedding_user = torch.nn.Embedding(NUM_USERS, self.embedding_dim)
        self.embedding_movie = torch.nn.Embedding(NUM_MOVIES, self.embedding_dim)

        self.dropout = torch.nn.Dropout(0.05)

        self.transform_u1 = KAN([self.embedding_dim, self.hidden_dim, self.embedding_dim])
        self.transform_u2 = KAN([self.embedding_dim, self.hidden_dim, self.embedding_dim])

        self.transform_m1 = KAN([self.embedding_dim, self.hidden_dim, self.embedding_dim])
        self.transform_m2 = KAN([self.embedding_dim, self.hidden_dim, self.embedding_dim])

        self.fc = torch.nn.Linear(3 * self.embedding_dim, 1)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("embedding_dim", int),
                ("hidden_dim", int),
            ],
        )

        self.embedding_dim = hyperparameters["embedding_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]

        if self.embedding_dim <= 0:
            raise ModelInitializationError("embedding_dim", "Parameter should be positive")
        if self.hidden_dim <= 0:
            raise ModelInitializationError("hidden_dim", "Parameter should be positive")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        users = inputs[:, 0]
        movies = inputs[:, 1]

        user_embedding = self.embedding_user(users)
        movie_embedding = self.embedding_movie(movies)

        u = self.dropout(user_embedding)
        m = self.dropout(movie_embedding)

        u1 = self.transform_u1(u)
        m1 = self.transform_m1(m)

        u1 = u1 + user_embedding
        m1 = m1 + movie_embedding

        u2 = self.transform_u2(u)
        m2 = self.transform_m2(m)

        out_combined = torch.mul(u2, m2)

        features = torch.cat([out_combined, u1, m1], dim=1)
        features = self.dropout(features)

        return self.fc(features)
