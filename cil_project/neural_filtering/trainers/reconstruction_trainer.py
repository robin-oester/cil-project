import logging
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.evaluators import AbstractEvaluator, ReconstructionEvaluator
from cil_project.neural_filtering.models import AbstractModel
from cil_project.utils import masked_mse
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .abstract_trainer import AbstractTrainer

logger = logging.getLogger(__name__)

CHECKPOINT_GRANULARITY = 50


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

    # pylint: disable=too-many-locals
    def train(self, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int) -> None:
        # TODO(#21): Implement other imputation methods than target mean.
        train_matrix = dataset.get_data_matrix(0)
        train_mask = dataset.get_data_matrix_mask()

        train_data_tensor = torch.from_numpy(train_matrix).to(self.device)
        train_mask_tensor = torch.from_numpy(train_mask).to(self.device)

        dataloader = DataLoader(
            TensorDataset(train_data_tensor, train_mask_tensor), batch_size=self.batch_size, shuffle=True
        )

        evaluator: Optional[AbstractEvaluator] = None
        if val_dataset is not None:
            evaluator = ReconstructionEvaluator(self.model, self.batch_size, dataset, val_dataset, self.device)

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

            val_loss: Optional[float] = None
            if evaluator is not None:
                val_loss = evaluator.evaluate()

            self._log_epoch_information(target_epoch, avg_train_loss, val_loss)

            if (self.current_epoch + 1) % CHECKPOINT_GRANULARITY == 0:
                self.save_state()

            if self.scheduler is not None:
                self.scheduler.step()
            self.current_epoch += 1

        logger.info(
            f"Finished training of model "
            f"{model_class_name} with best validation loss {self.best_val_loss:.4f}."
        )
