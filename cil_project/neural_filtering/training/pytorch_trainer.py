import logging
import pathlib
from datetime import datetime
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.models import AbstractModel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PyTorchTrainer:
    """
    Class used to train neural networks.
    """

    def __init__(
        self,
        model: AbstractModel,
        checkpoint_dir: pathlib.Path,
        batch_size: int,
        optimizer: Optimizer,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the trainer.

        :param model: the model to train.
        :param checkpoint_dir: path to the directory, in which to store checkpoints (must not exist).
        :param batch_size: batch size used during training.
        :param optimizer: optimizer used for training.
        :param device: (optional) device on which to run training.
        """

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(exist_ok=False)
        self.checkpoint_dir = checkpoint_dir

        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.current_epoch = 0

    def save_state(self) -> None:
        """
        Stores the current training state.
        """
        dict_to_save = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparameters": self.model.hyperparameters,
            "epoch": self.current_epoch + 1,
        }

        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        model_class_name = self.model.__class__.__name__
        epoch = self.current_epoch + 1
        torch.save(dict_to_save, self.checkpoint_dir / f"{model_class_name}_{epoch}_{current_timestamp}.pi")

        logger.info(f"Stored model {model_class_name} in epoch {epoch}.")

    def load_checkpoint(self, checkpoint_name: str) -> None:
        """
        Loads a training checkpoint given the file name.

        :param checkpoint_name: file name in which the checkpoint is stored.
        """

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        assert checkpoint_path, "Checkpoint path needs to point to a file"

        loaded_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(loaded_dict["model"])
        self.optimizer.load_state_dict(loaded_dict["optimizer"])
        self.current_epoch = loaded_dict["epoch"]

        model_class_name = self.model.__class__.__name__
        logger.info(f"Loaded checkpoint of model {model_class_name} in epoch {self.current_epoch}.")

    # pylint: disable=too-many-locals
    def train(self, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int) -> None:
        """
        Starts training the model.

        :param dataset: dataset, on which the model is trained.
        :param val_dataset: (optional) validation dataset to track test error.
        :param num_epochs: number of epochs.
        """

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        target_epoch = self.current_epoch + num_epochs
        model_class_name = self.model.__class__.__name__
        logger.info(f"Starting training of model {model_class_name} in epoch {self.current_epoch+1}/{target_epoch}.")
        while self.current_epoch < target_epoch:
            # Set the model to training mode
            self.model.train()

            epoch_loss = 0.0
            with tqdm(dataloader, unit="batches") as tepoch:
                tepoch.set_description(f"Epoch {self.current_epoch + 1}/{target_epoch}")

                inputs: torch.Tensor
                targets: torch.Tensor
                for inputs, targets in tepoch:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            if (self.current_epoch + 1) % 5 == 0:
                self.save_state()

            if val_loader is not None:
                # Validation phase
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                        epoch_val_loss += loss.item()

                avg_val_loss = epoch_val_loss / len(val_loader)

                logger.info(
                    f"Epoch [{self.current_epoch + 1}/{target_epoch}], Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch [{self.current_epoch + 1}/{target_epoch}], Train Loss: {avg_train_loss:.4f}")

            self.current_epoch += 1
