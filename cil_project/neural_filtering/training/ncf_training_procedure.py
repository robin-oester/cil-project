import argparse
import logging
import pathlib

import torch
from cil_project.dataset import BalancedSplit, RatingsDataset, TargetNormalization
from cil_project.neural_filtering.models import NCFBaseline
from cil_project.neural_filtering.training import PyTorchTrainer
from torch.optim import SGD

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_USERS = 10_000
NUM_MOVIES = 1_000
ALPHA = 0.5


class NCFTrainingProcedure:
    """
    Procedure used to train the NCF model.
    """

    def __init__(self, predictive_factor: int, batch_size: int, dataset: RatingsDataset, gmf_name: str, mlp_name: str):
        """
        Initializes the procedure.

        :param predictive_factor: the expressiveness of the model (must match with pretrained models).
        :param batch_size: batch size used for training.
        :param dataset: the dataset on which we train.
        :param gmf_name: name of the checkpoint used to initialize GMF weights.
        :param mlp_name: name of the checkpoint used to initialize MLP weights.
        """

        self.batch_size = batch_size
        self.dataset = dataset

        hyperparameters = {
            "num_users": NUM_USERS,
            "num_movies": NUM_MOVIES,
            "predictive_factor": predictive_factor,
            "alpha": ALPHA,
        }
        self.model = NCFBaseline(hyperparameters)

        current_path = pathlib.Path(__file__).resolve().parent
        checkpoint_path = current_path / "../checkpoints"

        # set the pretrained weights
        loaded_dict: dict = torch.load(checkpoint_path / gmf_name)["model"]
        loaded_dict.update(torch.load(checkpoint_path / mlp_name)["model"])

        self.model.load_state_dict(loaded_dict)

        # optimize using SGD
        optimizer = SGD(self.model.parameters(), lr=0.01)
        self.trainer = PyTorchTrainer(self.model, checkpoint_path, batch_size, optimizer)

    def start_training(self) -> None:
        splitter = BalancedSplit(0.9, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)
        test_dataset.set_dataset_statistics(train_dataset)

        train_dataset.normalize(TargetNormalization.TO_TANH_RANGE)
        test_dataset.normalize(TargetNormalization.TO_TANH_RANGE)

        self.trainer.train(train_dataset, test_dataset, 30)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NCF with the specified predictive factor, batch size and the pretrained models."
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

    parser.add_argument("--gmf", type=str, required=True, help="Name of the pretrained model file of GMF.")

    parser.add_argument("--mlp", type=str, required=True, help="Name of the pretrained model file of MLP.")

    args = parser.parse_args()

    predictive_factor: int = args.predictive_factor
    batch_size: int = args.batch_size
    gmf_name: str = args.gmf
    mlp_name: str = args.mlp

    logger.info(
        f"Initialized the training procedure for NCF with predictive factor {predictive_factor} "
        f"and batch size {batch_size}."
    )

    current_path = pathlib.Path(__file__).resolve().parent
    dataset = RatingsDataset.deserialize(current_path / "../../../data/serialized_ratings.npz")
    training_procedure = NCFTrainingProcedure(predictive_factor, batch_size, dataset, gmf_name, mlp_name)
    training_procedure.start_training()


if __name__ == "__main__":
    main()
