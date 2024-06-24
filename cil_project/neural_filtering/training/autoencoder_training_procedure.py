import argparse
import logging

from cil_project.dataset import BalancedSplit, RatingsDataset
from cil_project.neural_filtering.models.autoencoder import Autoencoder
from cil_project.neural_filtering.trainers import ReconstructionTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, NUM_MOVIES
from torch import optim
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train the autoencoder with the specified encoding size and batch size.
Typical usage:

./autoencoder_training_procedure.py --encoding_size <encoding_size> --batch_size <batch_size> [--dataset <dataset>]
"""

LEARNING_RATE = 0.005
NUM_EPOCHS = 120


class AutoencoderTrainingProcedure:
    """
    Class used to train the autoencoder.
    """

    def __init__(self, encoding_size: int, batch_size: int, dataset: RatingsDataset) -> None:
        self.batch_size = batch_size
        self.dataset = dataset

        hyperparameters = {"num_movies": NUM_MOVIES, "encoding_size": encoding_size}
        model = Autoencoder(hyperparameters)

        # initialize the trainer
        optimizer = Adam(model.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.trainer = ReconstructionTrainer(model, batch_size, optimizer, scheduler, verbose=False)

    def start_training(self) -> None:
        splitter = BalancedSplit(0.95, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)

        self.trainer.train(train_dataset, test_dataset, NUM_EPOCHS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoencoder with the specified encoding size and batch size.")

    parser.add_argument(
        "--encoding_size",
        type=int,
        choices=[16, 32, 64],
        required=True,
        help="The encoding size of the model, which must be one of 5, 10, 15 or 30.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[16, 32, 64, 128],
        required=True,
        help="The batch size used for training. Must be one of 16, 32, 64, 128.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=RatingsDataset.get_available_dataset_names(),
        required=False,
        default=FULL_SERIALIZED_DATASET_NAME,
        help="The name of the dataset.",
    )

    args = parser.parse_args()

    encoding_size: int = args.encoding_size
    batch_size: int = args.batch_size
    dataset_name: str = args.dataset

    logger.info(
        f"Initialized the training procedure for the autoencoder with encoding size {encoding_size} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = AutoencoderTrainingProcedure(encoding_size, batch_size, dataset)
    training_procedure.start_training()


if __name__ == "__main__":
    main()
