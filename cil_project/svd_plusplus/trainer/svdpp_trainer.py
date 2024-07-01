import logging
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from cil_project.svd_plusplus.evaluators import AbstractEvaluator, SVDPPEvaluator
from cil_project.svd_plusplus.model import AbstractModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_trainer import AbstractTrainer

logger = logging.getLogger(__name__)

CHECKPOINT_GRANULARITY = 5


class SVDPPTrainer(AbstractTrainer):
    """
    Class used to train svdpp.
    """

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
        self.validation_loss: Optional[float] = 0.0

    # pylint: disable=too-many-locals
    def train(self, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int) -> None:

        # Initialize mu, bu, bi, and y
        self.model.compute_mu_bu_bi_y(dataset.get_data_matrix(), dataset.get_data_matrix_mask())

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        loss_func = torch.nn.MSELoss(reduction="none")

        evaluator: Optional[AbstractEvaluator] = None
        if val_dataset is not None:
            evaluator = SVDPPEvaluator(self.model, self.batch_size, dataset, val_dataset, self.device)

        target_epoch = self.current_epoch + num_epochs
        model_class_name = self.model.__class__.__name__
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
                    # ------------ Start loss computation over the batch
                    base_loss = loss_func(outputs, targets)
                    users = inputs[:, 0]
                    items = inputs[:, 1]
                    p_u = self.model.p(users)
                    q_i = self.model.q(items)
                    reg_pu = self.model.lam * torch.norm(p_u, 2, dim=1) ** 2
                    reg_qi = self.model.lam * torch.norm(q_i, 2, dim=1) ** 2
                    total_loss = base_loss + reg_pu + reg_qi
                    total_loss = total_loss.mean()
                    # ------------ End loss computation over the batch
                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            if (self.current_epoch + 1) % CHECKPOINT_GRANULARITY == 0:
                self.save_state()

            val_loss: Optional[float] = None
            if evaluator is not None:
                val_loss = evaluator.evaluate()
                self.validation_loss = val_loss

            self._log_epoch_information(target_epoch, avg_train_loss, val_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            self.current_epoch += 1

        logger.info(f"Finished training of model {model_class_name}.")
