import logging
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from cil_project.svd_plusplus.model import AbstractModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent / "checkpoints"

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):
    """
    Abstract class used to train neural networks.
    """

    def __init__(
        self,
        model: AbstractModel,
        batch_size: int,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: Optional[str],
        verbose: bool,
    ) -> None:
        """
        Initializes the trainer.

        :param model: the model to train.
        :param batch_size: batch size used during training.
        :param optimizer: optimizer used for training.
        :param scheduler: (optional) scheduler for the learning rate.
        :param device: (optional) device on which to run training.
        :param verbose: whether to print training information.
        """
        self.model = model
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # check if optimizer in scheduler corresponds to the optimizer in the model
        if self.scheduler is not None:
            assert (
                self.scheduler.optimizer == self.optimizer
            ), "Optimizer in scheduler must match the optimizer in the model."

        self.model.to(self.device)
        self.verbose = verbose
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def save_state(self) -> None:
        """
        Stores the current training state.
        """

        dict_to_save = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparameters": self.model.hyperparameters,
            "epoch": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
        }

        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        model_class_name = self.model.__class__.__name__
        epoch = self.current_epoch + 1
        torch.save(dict_to_save, CHECKPOINT_PATH / f"{model_class_name}_{epoch}_{current_timestamp}.pkl")

        logger.info(f"Stored model {model_class_name} in epoch {epoch}.")

    def load_checkpoint(self, checkpoint_name: str) -> None:
        """
        Loads a training checkpoint given the file name.

        :param checkpoint_name: file name in which the checkpoint is stored.
        """

        checkpoint_path = CHECKPOINT_PATH / checkpoint_name
        assert checkpoint_path, "Checkpoint path needs to point to a file"

        loaded_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(loaded_dict["model"])
        self.optimizer.load_state_dict(loaded_dict["optimizer"])
        self.current_epoch = loaded_dict["epoch"]
        self.best_val_loss = loaded_dict["best_val_loss"]

        model_class_name = self.model.__class__.__name__
        logger.info(f"Loaded checkpoint of model {model_class_name} in epoch {self.current_epoch}.")

    def _log_epoch_information(self, target_epoch: int, training_loss: float, validation_loss: Optional[float]) -> None:
        """
        Logs the training loss and validation loss (if available) at the end of an epoch.
        Additionally, checks if the validation loss is the best so far and stores it if so.

        :param target_epoch: how many epochs to train in total.
        :param training_loss: loss on the training set.
        :param validation_loss: (optional) loss on the validation set.
        """

        if validation_loss is not None:
            if self.verbose:
                logger.info(
                    f"Epoch [{self.current_epoch + 1}/{target_epoch}], Train Loss: {training_loss:.4f}, "
                    f"Validation Loss: {validation_loss:.4f} {'(best)' if validation_loss < self.best_val_loss else ''}"
                )
            elif validation_loss < self.best_val_loss:
                logger.info(
                    f"Epoch [{self.current_epoch + 1}/{target_epoch}], best validation loss: {validation_loss:.4f}"
                )

            self.best_val_loss = self.best_val_loss if validation_loss >= self.best_val_loss else validation_loss

        elif self.verbose:
            logger.info(f"Epoch [{self.current_epoch + 1}/{target_epoch}], Train Loss: {training_loss:.4f}")

    @abstractmethod
    def train(self, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int) -> None:
        """
        Starts training the model.

        :param dataset: dataset, on which the model is trained.
        :param val_dataset: (optional) validation dataset to track test error.
        :param num_epochs: number of epochs.
        """

        raise NotImplementedError()
