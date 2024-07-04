import argparse
import logging
from typing import Optional

from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.models.ncf_baseline import NCFGMFModel, NCFMLPModel
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
This script is used to pretrain the NCF components with the specified predictive factor and batch size.
Typical usage:

./ncf_pretraining_procedure.py --predictive_factor <predictive_factor> --batch_size <batch_size>
                               --model_type <model_type> [--dataset <dataset>] [--val_dataset <dataset>]
"""

# learning constants
LEARNING_RATE = 0.001
NUM_EPOCHS = 100


class NCFPretrainingProcedure:
    """
    Class used to perform pretraining of the NCF components.
    """

    def __init__(self, predictive_factor: int, batch_size: int) -> None:
        """
        Initialize the procedure to train the model.

        :param predictive_factor: the model capability.
        :param batch_size: the batch size used for training.
        """

        self.batch_size = batch_size
        self.hyperparameters = {"predictive_factor": predictive_factor}

    def start_training(
        self, model_type: str, dataset: RatingsDataset, val_dataset: Optional[RatingsDataset], num_epochs: int
    ) -> None:
        # select the correct model
        if model_type == "gmf":
            model = NCFGMFModel(self.hyperparameters)
        else:
            model = NCFMLPModel(self.hyperparameters)

        # initialize the trainer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        trainer = RatingTrainer(model, self.batch_size, optimizer)

        # optionally, normalize the training dataset
        # dataset.normalize(TargetNormalization.TO_TANH_RANGE)

        try:
            trainer.train(dataset, val_dataset, num_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain GMF/MLP with the specified predictive factor and batch size."
    )

    parser.add_argument(
        "--predictive_factor",
        type=int,
        choices=[8, 16, 32, 64, 128],
        required=True,
        help="The predictive factor for the model, which must be one of 8, 16, 32, 64, 128.",
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
    model_type: str = args.model_type
    dataset_name: str = args.dataset
    val_dataset_name: str = args.val_dataset

    logger.info(
        f"Initialized the pretraining procedure for {model_type} with predictive factor {predictive_factor} "
        f"and batch size {batch_size} on dataset '{dataset_name}' "
        f"{'without' if val_dataset_name is None else 'with'} validation."
    )

    dataset = RatingsDataset.load(dataset_name)

    val_dataset: Optional[RatingsDataset] = None
    if val_dataset_name is not None:
        val_dataset = RatingsDataset.load(val_dataset_name)
    training_procedure = NCFPretrainingProcedure(predictive_factor, batch_size)
    training_procedure.start_training(model_type, dataset, val_dataset, NUM_EPOCHS)


if __name__ == "__main__":
    main()
