import logging
import pathlib
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_trainer import AbstractTrainer

logger = logging.getLogger(__name__)


class RatingTrainer(AbstractTrainer):
    """
    Class used to train neural networks.
    """

    # pylint: disable=too-many-locals
    def train(
        self, dataset: RatingsDataset, num_epochs: int, checkpoint_granularity: Optional[int] = None
    ) -> Optional[float]:
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_func = torch.nn.MSELoss()

        target_epoch = self.current_epoch + num_epochs
        model_class_name = self.model.__class__.__name__
        last_val_loss: Optional[float] = None
        last_checkpoint: Optional[pathlib.Path] = None
        logger.info(f"Starting training of model {model_class_name} in epoch {self.current_epoch+1}/{target_epoch}.")

        while self.current_epoch < target_epoch:
            # Set the model to training mode
            self.model.train()

            epoch_loss = 0.0
            with tqdm(dataloader, unit="batches", disable=not self.verbose) as tepoch:
                tepoch.set_description(f"Epoch {self.current_epoch + 1}/{target_epoch}")

                inputs: torch.Tensor
                targets: torch.Tensor
                for inputs, targets in tepoch:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = loss_func(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)

            if self.evaluator is not None and self.evaluator.val_dataset is not None:
                last_val_loss = self.evaluator.evaluate()

            self._log_epoch_information(target_epoch, avg_train_loss, last_val_loss)

            if self.must_save_checkpoint(target_epoch, checkpoint_granularity):
                last_checkpoint = self.save_state()

            if self.scheduler is not None:
                self.scheduler.step()
            self.current_epoch += 1

        logger.info(f"Finished training of model {model_class_name} for {num_epochs} epochs.")
        if last_val_loss is not None:
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}. Last validation loss: {last_val_loss:.4f}.")
        if last_checkpoint is not None:
            logger.info(f"Stored last checkpoint to '{last_checkpoint.name}'.")

        return last_val_loss
