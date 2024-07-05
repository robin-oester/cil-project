import argparse
import logging
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.models import NCFBaseline
from cil_project.neural_filtering.trainers import RatingTrainer
from cil_project.utils import CHECKPOINT_PATH, FULL_SERIALIZED_DATASET_NAME
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
                            --gmf <gmf_name> --mlp <mlp_name> [--dataset <dataset>] [--val_dataset <dataset>]
"""

# learning constants
ALPHA = 0.5
LEARNING_RATE = 0.01
NUM_EPOCHS = 100


class NCFTrainingProcedure:
    """
    Procedure used to train the NCF model.
    """

    def __init__(self, predictive_factor: int, batch_size: int):
        """
        Initializes the procedure.

        :param predictive_factor: the expressiveness of the model (must match with pretrained models).
        :param batch_size: batch size used for training.
        """

        self.batch_size = batch_size
        self.hyperparameters = {
            "predictive_factor": predictive_factor,
            "alpha": ALPHA,
        }

    def start_training(
        self,
        gmf_name: str,
        mlp_name: str,
        dataset: RatingsDataset,
        val_dataset: Optional[RatingsDataset],
        num_epochs: int,
    ) -> None:
        model = NCFBaseline(self.hyperparameters)

        # load pretrained models and initialize weights of full model
        loaded_dict: dict = torch.load(CHECKPOINT_PATH / gmf_name)["model"]
        loaded_dict.update(torch.load(CHECKPOINT_PATH / mlp_name)["model"])
        model.load_state_dict(loaded_dict)

        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
        trainer = RatingTrainer(model, self.batch_size, optimizer)

        # optionally, normalize the training dataset
        # dataset.normalize(TargetNormalization.TO_TANH_RANGE)

        try:
            trainer.train(dataset, val_dataset, num_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NCF with the specified predictive factor, batch size and the pretrained models."
    )

    parser.add_argument(
        "--predictive_factor",
        type=int,
        choices=[8, 16, 32, 64, 128],
        required=True,
        help="The predictive factor for the model, which must be one of 8, 16, 32, 64 or 128.",
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

    parser.add_argument(
        "--val_dataset",
        type=str,
        choices=RatingsDataset.get_available_dataset_names(),
        required=False,
        default=None,
        help="The name of the validation dataset.",
    )

    args = parser.parse_args()

    predictive_factor: int = args.predictive_factor
    batch_size: int = args.batch_size
    gmf_name: str = args.gmf
    mlp_name: str = args.mlp
    dataset_name: str = args.dataset
    val_dataset_name: str = args.val_dataset

    logger.info(
        f"Initialized the training procedure for NCF with predictive factor {predictive_factor} "
        f"and batch size {batch_size} on dataset '{dataset_name}' "
        f"{'without' if val_dataset_name is None else 'with'} validation."
    )

    dataset = RatingsDataset.load(dataset_name)
    val_dataset: Optional[RatingsDataset] = None
    if val_dataset_name is not None:
        val_dataset = RatingsDataset.load(val_dataset_name)

    training_procedure = NCFTrainingProcedure(predictive_factor, batch_size)
    training_procedure.start_training(gmf_name, mlp_name, dataset, val_dataset, NUM_EPOCHS)


if __name__ == "__main__":
    main()
