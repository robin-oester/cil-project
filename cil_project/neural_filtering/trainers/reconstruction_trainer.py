import logging
from typing import Optional

import numpy as np
import torch
from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.models import AbstractModel
from cil_project.utils import MAX_RATING, MIN_RATING, NUM_MOVIES, NUM_USERS, masked_mse, masked_rmse
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .abstract_trainer import AbstractTrainer

logger = logging.getLogger(__name__)

CHECKPOINT_GRANULARITY = 20


class ReconstructionTrainer(AbstractTrainer):

    def __init__(
        self,
        model: AbstractModel,
        batch_size: int,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(model, batch_size, optimizer, scheduler, device, verbose)

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

    # pylint: disable=too-many-locals
    def train(self, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int) -> None:
        # imputes training dataset with target mean
        train_matrix = dataset.get_data_matrix(dataset.get_target_mean())
        train_mask = dataset.get_data_matrix_mask()

        train_data_tensor = torch.tensor(train_matrix, device=self.device)
        train_mask_tensor = torch.tensor(train_mask, device=self.device)

        dataloader = DataLoader(
            TensorDataset(train_data_tensor, train_mask_tensor), batch_size=self.batch_size, shuffle=True
        )

        test_matrix = None
        test_mask = None
        if val_dataset is not None:
            test_matrix = val_dataset.get_data_matrix()
            test_mask = np.where(test_matrix != 0, 1, 0)

        target_epoch = self.current_epoch + num_epochs
        model_class_name = self.model.__class__.__name__
        logger.info(f"Starting training of model {model_class_name} in epoch {self.current_epoch + 1}/{target_epoch}.")

        while self.current_epoch < target_epoch:
            # Set the model to training mode
            self.model.train()

            epoch_loss = 0.0
            with tqdm(dataloader, unit="batches", disable=not self.verbose) as tepoch:
                tepoch.set_description(f"Epoch {self.current_epoch + 1}/{num_epochs}")

                inputs: torch.Tensor
                masks: torch.Tensor
                for inputs, masks in tepoch:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = masked_mse(outputs, inputs, masks)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            if (self.current_epoch + 1) % CHECKPOINT_GRANULARITY == 0:
                self.save_state()

            val_loss: Optional[float] = None
            if val_dataset is not None:
                self.model.eval()
                with torch.no_grad():
                    reconstructed_matrix = self._reconstruct_whole_matrix(train_data_tensor)

                    reconstructed_matrix = np.clip(reconstructed_matrix, MIN_RATING, MAX_RATING)
                    val_loss = masked_rmse(test_matrix, reconstructed_matrix, test_mask)

            self._log_epoch_information(target_epoch, avg_train_loss, val_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            self.current_epoch += 1

        logger.info(f"Finished training of model {model_class_name}.")
        self.save_state()
