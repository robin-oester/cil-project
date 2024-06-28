import logging

from cil_project.dataset import BalancedSplit, RatingsDataset
from cil_project.svd_plusplus.model import SVDPP
from cil_project.svd_plusplus.trainer import SVDPPTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from torch import optim
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train svdpp.
Typical usage: python3 cil_project/../svdpp_training_procedure.py
"""

LEARNING_RATE = 0.05
NUM_EPOCHS = 5


class SVDPPTrainingProcedure:
    """
    Class used to train SVDPP.
    """

    def __init__(self, hyperparameters: dict, batch_size: int, dataset: RatingsDataset) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        model = SVDPP(hyperparameters)
        # initialize the trainer
        optimizer = Adam(model.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

        self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

    def start_training(self) -> None:
        splitter = BalancedSplit(0.95, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)

        try:
            self.trainer.train(train_dataset, test_dataset, NUM_EPOCHS)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:

    # hyperparameters
    hyperparameters = {"nr_factors": 190, "lam": 0.001, "lam1": 10.0, "lam2": 25.0, "lam3": 10.0}

    batch_size = 4096  # works fine
    dataset_name = FULL_SERIALIZED_DATASET_NAME

    logger.info(
        f"Initialized the training procedure for svdpp with hyperparameters {hyperparameters} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = SVDPPTrainingProcedure(hyperparameters, batch_size, dataset)
    training_procedure.start_training()


if __name__ == "__main__":
    main()
