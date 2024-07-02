import argparse
import logging

from cil_project.dataset import BalancedSplit, RatingsDataset, TargetNormalization
from cil_project.neural_filtering.models.autoencoder import Autoencoder
from cil_project.neural_filtering.trainers import ReconstructionTrainer
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
This script is used to train the autoencoder with the specified encoding size, dropout probability and batch size.
Typical usage:

./autoencoder_training_procedure.py --encoding_size <encoding_size> --dropout <probability> --batch_size <batch_size>
"""

# learning constants
LEARNING_RATE = 0.025
GAMMA = 0.992
NUM_EPOCHS = 1000


class AutoencoderTrainingProcedure:
    """
    Class used to train the autoencoder.
    """

    def __init__(self, encoding_size: int, p_dropout: float, batch_size: int) -> None:
        self.batch_size = batch_size

        self.model_hyperparameters = {"encoding_size": encoding_size, "p_dropout": p_dropout}

    def start_training(self, num_epochs: int) -> None:
        model = Autoencoder(self.model_hyperparameters)

        # initialize the trainer
        optimizer = Adam(model.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
        trainer = ReconstructionTrainer(model, self.batch_size, optimizer, scheduler)

        dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
        splitter = BalancedSplit(0.95, True)

        train_idx, test_idx = splitter.split(dataset)

        train_dataset = dataset.get_split(train_idx)
        test_dataset = dataset.get_split(test_idx)

        # optionally, normalize the training dataset
        train_dataset.normalize(TargetNormalization.TO_TANH_RANGE)

        try:
            trainer.train(train_dataset, test_dataset, num_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoencoder with the specified encoding size and batch size.")

    parser.add_argument(
        "--encoding_size",
        type=int,
        choices=[16, 32, 64, 128],
        required=True,
        help="The encoding size of the model, which must be one of 16, 32, 64, 128.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        required=True,
        help="The dropout probability of the neurons.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[32, 64, 128, 256, 512],
        required=True,
        help="The batch size used for training. Must be one of 32, 64, 128, 256, 512.",
    )

    args = parser.parse_args()

    encoding_size: int = args.encoding_size
    p_dropout: int = args.dropout
    batch_size: int = args.batch_size

    logger.info(
        f"Initialized the training procedure for the autoencoder with encoding size {encoding_size}, "
        f"dropout probability {p_dropout} and batch size {batch_size}."
    )

    training_procedure = AutoencoderTrainingProcedure(encoding_size, p_dropout, batch_size)
    training_procedure.start_training(NUM_EPOCHS)


if __name__ == "__main__":
    main()
