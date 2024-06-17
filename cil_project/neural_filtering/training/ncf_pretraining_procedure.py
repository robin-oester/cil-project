import argparse
import logging
import pathlib

from cil_project.dataset import BalancedSplit, RatingsDataset, TargetNormalization
from cil_project.neural_filtering.models.ncf_baseline import NCFGMFModel, NCFMLPModel
from cil_project.neural_filtering.training import PyTorchTrainer
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

# dataset constants
NUM_USERS = 10_000
NUM_MOVIES = 1_000


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
            self.model = NCFGMFModel(hyperparameters)
        else:
            self.model = NCFMLPModel(hyperparameters)

        # choose checkpoint directory
        current_path = pathlib.Path(__file__).resolve().parent
        checkpoint_path = current_path / "../checkpoints"

        # initialize the trainer
        optimizer = Adam(self.model.parameters(), lr=0.001)
        self.trainer = PyTorchTrainer(self.model, checkpoint_path, batch_size, optimizer)

    def start_training(self, num_epochs: int) -> None:
        splitter = BalancedSplit(0.9, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)

        # update test dataset statistics to reflect the training dataset
        test_dataset.set_dataset_statistics(train_dataset)

        train_dataset.normalize(TargetNormalization.TO_TANH_RANGE)
        test_dataset.normalize(TargetNormalization.TO_TANH_RANGE)

        self.trainer.train(train_dataset, test_dataset, num_epochs)


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

    args = parser.parse_args()

    predictive_factor: int = args.predictive_factor
    batch_size: int = args.batch_size
    model_type: str = args.model_type

    logger.info(
        f"Initialized the pretraining procedure for {model_type} with predictive factor {predictive_factor} "
        f"and batch size {batch_size}."
    )

    current_path = pathlib.Path(__file__).resolve().parent
    dataset = RatingsDataset.deserialize(current_path / "../../../data/serialized_ratings.npz")
    training_procedure = NCFPretrainingProcedure(model_type, predictive_factor, batch_size, dataset)
    training_procedure.start_training(30)


if __name__ == "__main__":
    main()
