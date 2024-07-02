import argparse
import logging

from cil_project.dataset import BalancedSplit, RatingsDataset, TargetNormalization
from cil_project.neural_filtering.models import KANNCF
from cil_project.neural_filtering.trainers import RatingTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train the KAN-based NCF with the specified embedding size, hidden dimension and batch size.
Typical usage:

./kan_ncf_procedure --embedding <embedding_dim> --hidden <hidden_dim> --batch_size <batch_size>
"""

# learning constants
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4


class KANNCFProcedure:
    """
    Class used to perform training of the KAN-based NCF model.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, batch_size: int) -> None:
        """
        Initialize the procedure to train the model.

        :param embedding_dim: the dimension of the embeddings.
        :param hidden_dim: the dimension of the hidden layers.
        :param batch_size: the batch size used for training.
        """

        self.batch_size = batch_size
        self.hyperparameters = {"embedding_dim": embedding_dim, "hidden_dim": hidden_dim}

    def start_training(self, num_epochs: int) -> None:
        model = KANNCF(self.hyperparameters)

        # initialize the trainer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        trainer = RatingTrainer(model, self.batch_size, optimizer)

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
    parser = argparse.ArgumentParser(description="Train KAN-based NCF with the specified parameters.")

    parser.add_argument(
        "--embedding",
        type=int,
        choices=[32, 64, 128, 256, 512],
        required=True,
        help="The embedding dimension of the model. Must be one of 32, 64, 128, 256, 512.",
    )

    parser.add_argument(
        "--hidden",
        type=int,
        choices=[8, 16, 32, 64, 128, 256],
        required=True,
        help="The hidden dimension of the model. Must be one of 8, 16, 32, 64, 128, 256.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[64, 128, 256, 512],
        required=True,
        help="The batch size used for training. Must be one of 64, 128, 256, 512.",
    )

    args = parser.parse_args()

    embedding_dim: int = args.embedding
    hidden_dim: int = args.hidden
    batch_size: int = args.batch_size

    logger.info(
        f"Initialized the procedure for training KAN-based NCF with embedding dimension {embedding_dim}, hidden "
        f"dimension {hidden_dim} and batch size {batch_size}."
    )

    training_procedure = KANNCFProcedure(embedding_dim, hidden_dim, batch_size)
    training_procedure.start_training(NUM_EPOCHS)


if __name__ == "__main__":
    main()
