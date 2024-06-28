import argparse
import logging

from cil_project.dataset import BalancedSplit, RatingsDataset
from cil_project.neural_filtering.models.ncf_baseline import NCFGMFModel, NCFMLPModel
from cil_project.neural_filtering.trainers import RatingTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, NUM_MOVIES, NUM_USERS
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to pretrain the NCF components with the specified predictive factor and batch size.
Typical usage:

./ncf_pretraining_procedure.py --predictive_factor <predictive_factor> --batch_size <batch_size>
                               --model_type <model_type> [--dataset <dataset>]
"""

# dataset constants
LEARNING_RATE = 0.001
NUM_EPOCHS = 100


class NCFPretrainingProcedure:
    """
    Class used to perform pretraining of the NCF components.
    """

    def __init__(self, model_type: str, predictive_factor: int, batch_size: int, dataset: RatingsDataset) -> None:
        """
        Initialize the procedure to train the model.

        :param model_type: one of: {'gmf', 'mlp'}.
        :param predictive_factor: the model capability.
        :param batch_size: the batch size used for training.
        :param dataset: the dataset on which we train.
        """

        self.batch_size = batch_size
        self.dataset = dataset

        hyperparameters = {"num_users": NUM_USERS, "num_movies": NUM_MOVIES, "predictive_factor": predictive_factor}

        # select the correct model
        if model_type == "gmf":
            model = NCFGMFModel(hyperparameters)
        else:
            model = NCFMLPModel(hyperparameters)

        # initialize the trainer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        self.trainer = RatingTrainer(model, batch_size, optimizer)

    def start_training(self, num_epochs: int) -> None:
        splitter = BalancedSplit(0.95, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)

        # train_dataset.normalize(TargetNormalization.TO_TANH_RANGE)  # target normalization

        try:
            self.trainer.train(train_dataset, test_dataset, num_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain GMF/MLP with the specified predictive factor and batch size."
    )

    parser.add_argument(
        "--predictive_factor",
        type=int,
        choices=[8, 16, 32, 64],
        required=True,
        help="The predictive factor for the model, which must be one of 8, 16, 32, or 64.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[64, 128, 256, 512],
        required=True,
        help="The batch size used for training. Must be one of 64, 128, 256, 512.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gmf", "mlp"],
        required=True,
        help="The model type, which must be either 'mlp' or 'gmf'.",
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

    predictive_factor: int = args.predictive_factor
    batch_size: int = args.batch_size
    model_type: str = args.model_type
    dataset_name: str = args.dataset

    logger.info(
        f"Initialized the pretraining procedure for {model_type} with predictive factor {predictive_factor} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = NCFPretrainingProcedure(model_type, predictive_factor, batch_size, dataset)
    training_procedure.start_training(NUM_EPOCHS)


if __name__ == "__main__":
    main()
