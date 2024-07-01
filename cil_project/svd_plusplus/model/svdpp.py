from typing import Any

import numpy as np
import torch
from cil_project.utils import NUM_MOVIES, NUM_USERS, ModelInitializationError, validate_parameter_types
from torch import nn

from .abstract_model import AbstractModel


class SVDPP(AbstractModel):
    """
    SVD++ model for collaborative filtering.
    """

    # pylint: disable=useless-parent-delegation
    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

    def _initialize_parameters(self, hyperparameters: dict[str, Any]) -> None:
        validate_parameter_types(
            hyperparameters,
            [
                ("nr_factors", int),
                ("lam", float),
                ("lam1", float),
                ("lam2", float),
                ("lam3", float),
            ],
        )
        self.nr_factors = hyperparameters["nr_factors"]
        self.lam = hyperparameters["lam"]
        self.lam1 = hyperparameters["lam1"]
        self.lam2 = hyperparameters["lam2"]
        self.lam3 = hyperparameters["lam3"]

        # Calculated using heuristics as soon as data is available
        self.mu = torch.tensor(0.0)
        self.bu = torch.zeros(NUM_USERS, requires_grad=False)
        self.bi = torch.zeros(NUM_MOVIES, requires_grad=False)
        self.y = torch.zeros(NUM_USERS, self.nr_factors, requires_grad=False)

        self.init_mean = 0.0
        self.init_std = 0.005

        # Parameters that are going to be learned
        # user factor matrix p
        self.p = nn.Embedding(NUM_USERS, self.nr_factors).float()
        nn.init.normal_(self.p.weight, mean=self.init_mean, std=self.init_std)
        # item factor matrix q
        self.q = nn.Embedding(NUM_MOVIES, self.nr_factors).float()
        nn.init.normal_(self.q.weight, mean=self.init_mean, std=self.init_std)

        if self.nr_factors <= 0:
            raise ModelInitializationError("nr_factors", "Parameter should be positive")

    # Article: New Collaborative Filtering Algorithms Based on SVD++ and Differential Privacy
    # https://doi.org/10.1155/2017/1975719
    # pylint: disable=too-many-locals, attribute-defined-outside-init
    def compute_mu_bu_bi_y(self, data_matrix: np.ndarray, data_matrix_mask: np.ndarray) -> None:
        """
        Compute global mean (mu), user bias (bu), item bias (bi) and implicit feedback (y) for the SVD++ model.
        """
        print("Calculating mu, bu, bi, and y...")
        # Compute the global mean (mu)
        data_matrix_mask = data_matrix_mask.astype(bool)
        mu = np.mean(data_matrix[data_matrix_mask])

        bi = np.zeros(NUM_MOVIES)
        # Compute item biases (bi)
        for i in range(NUM_MOVIES):
            indices = np.where(data_matrix_mask[:, i])[0]
            ratings_for_i = data_matrix[indices, i]
            # Compute bias for item i
            bi[i] = (ratings_for_i - mu).sum() / (self.lam1 + len(ratings_for_i))

        bu = np.zeros(NUM_USERS)
        # Compute user biases (bu)
        for u in range(NUM_USERS):
            indices = np.where(data_matrix_mask[u])[0]
            ratings_of_u = data_matrix[u, indices]
            # Compute bias for user u
            bu[u] = (ratings_of_u - mu - bi[indices]).sum() / (self.lam2 + len(ratings_of_u))

        # Compute the y matrix
        # Adapted from: https://github.com/opedal/cilsg
        y = np.zeros((NUM_USERS, self.nr_factors))
        _, s, vt = np.linalg.svd(data_matrix, full_matrices=False)
        d = np.diag(np.sqrt(s))
        v = d.dot(vt.T)
        v = v[:, : self.nr_factors]

        for u in range(NUM_USERS):
            rated_items_indices = np.where(data_matrix_mask[u])[0]
            for i in rated_items_indices:
                y[u, :] += v[i, :]
            y[u, :] /= (self.lam3 + len(rated_items_indices)) * np.sqrt(len(rated_items_indices))

        self.mu = torch.tensor(mu).float()
        self.bu = torch.from_numpy(bu).float()
        self.bi = torch.from_numpy(bi).float()
        self.y = torch.from_numpy(y).float()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        u = inputs[:, 0].long()
        i = inputs[:, 1].long()

        bu = self.bu[u]
        bi = self.bi[i]
        q_i = self.q(i)
        p_u = self.p(u)
        y_u = self.y[u, :]

        dot_product = (q_i * (p_u + y_u)).sum(dim=1)

        pred = self.mu + bu + bi + dot_product

        pred = pred.unsqueeze(1)
        return pred
