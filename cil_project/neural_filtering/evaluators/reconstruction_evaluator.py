from typing import Optional

import numpy as np
import torch
from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.evaluators.abstract_evaluator import AbstractEvaluator
from cil_project.neural_filtering.models import AbstractModel
from cil_project.utils import NUM_MOVIES, NUM_USERS


class ReconstructionEvaluator(AbstractEvaluator):
    """
    Evaluator for models that aim at reconstructing the whole prediction matrix.
    """

    def __init__(
        self,
        model: AbstractModel,
        batch_size: int,
        dataset: RatingsDataset,
        val_dataset: Optional[RatingsDataset],
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model, batch_size, dataset, val_dataset, device)

        # helpers for efficient evaluation
        if val_dataset is not None and not dataset.is_normalized():
            self.test_matrix = val_dataset.get_data_matrix()
            self.test_mask = np.where(self.test_matrix != 0, 1, 0)

        # TODO(#21): Implement other imputation methods than target mean.
        train_matrix = self.dataset.get_data_matrix(0)
        self.train_data_tensor = torch.tensor(train_matrix, device=self.device)

    def _reconstruct_whole_matrix(self, train_data_tensor: torch.Tensor) -> np.ndarray:
        """
        Reconstructs the whole matrix from the model. Make sure that the model is in evaluation mode.

        :param train_data_tensor: training dataset to evaluate the model.
        :return: the reconstructed matrix.
        """

        data_reconstructed = np.zeros((NUM_USERS, NUM_MOVIES))

        for i in range(0, NUM_USERS, self.batch_size):
            upper_bound = min(i + self.batch_size, NUM_USERS)
            data_reconstructed[i:upper_bound] = self.model(train_data_tensor[i:upper_bound]).detach().cpu().numpy()

        return data_reconstructed

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        predictions = np.empty((inputs.shape[0], 1), dtype=np.float32)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructed_matrix = self._reconstruct_whole_matrix(self.train_data_tensor)

        for idx, (user_id, movie_id) in enumerate(inputs):
            predictions[idx] = reconstructed_matrix[user_id, movie_id]

        return predictions
