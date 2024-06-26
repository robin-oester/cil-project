import argparse
import logging

import torch
from cil_project.dataset import BalancedSplit, RatingsDataset
from cil_project.neural_filtering.models import NCFBaseline
from cil_project.neural_filtering.trainers import RatingTrainer
from cil_project.utils import CHECKPOINT_PATH, FULL_SERIALIZED_DATASET_NAME, NUM_MOVIES, NUM_USERS
from torch.optim import SGD

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train the NCF model with the specified predictive factor and batch size.
It also takes the pretrained model names (as found in checkpoints) for GMF and MLP.
Typical usage:

./ncf_training_procedure.py --predictive_factor <predictive_factor> --batch_size <batch_size>
                            --gmf <gmf_name> --mlp <mlp_name> [--dataset <dataset>]
"""

ALPHA = 0.5
LEARNING_RATE = 0.01
NUM_EPOCHS = 100


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
        model = NCFBaseline(hyperparameters)

        # set the pretrained weights
        loaded_dict: dict = torch.load(CHECKPOINT_PATH / gmf_name)["model"]
        loaded_dict.update(torch.load(CHECKPOINT_PATH / mlp_name)["model"])
        model.load_state_dict(loaded_dict)

        # optimize using SGD
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

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
    gmf_name: str = args.gmf
    mlp_name: str = args.mlp
    dataset_name: str = args.dataset

    logger.info(
        f"Initialized the training procedure for NCF with predictive factor {predictive_factor} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = NCFTrainingProcedure(predictive_factor, batch_size, dataset, gmf_name, mlp_name)
    training_procedure.start_training(NUM_EPOCHS)


if __name__ == "__main__":
    main()
