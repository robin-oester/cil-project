import argparse
import logging
from typing import Optional

from cil_project.dataset import BalancedSplit, RatingsDataset, TargetNormalization
from cil_project.neural_filtering.models import NCFGMFModel, NCFMLPModel
from cil_project.neural_filtering.trainers import RatingTrainer
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
This script is used to pretrain the NCF components with the specified predictive factor and batch size.
Typical usage:

./ncf_combined_pretraining_procedure.py --predictive_factor <predictive_factor> --batch_size <batch_size>
                                        --model_type <model_type> [--dataset <dataset>] [--val_dataset <dataset>]
"""

# learning constants
NUM_EPOCHS = 50

LR_GMF = 1e-2
DECAY_GMF = 1e-5
GAMMA_GMF = 0.85

LR_MLP = 1e-3
DECAY_MLP = 1e-4
GAMMA_MLP = 0.85


class NCFCombinedPretrainingProcedure:
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

            optimizer = Adam(model.parameters(), lr=LR_GMF, weight_decay=DECAY_GMF)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA_GMF)
        else:
            model = NCFMLPModel(self.hyperparameters)

            optimizer = Adam(model.parameters(), lr=LR_MLP, weight_decay=DECAY_MLP)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA_MLP)

        # initialize the trainer
        trainer = RatingTrainer(model, self.batch_size, optimizer, scheduler)

        # optionally, normalize the training dataset
        dataset.normalize(TargetNormalization.BY_MOVIE)

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
        required=True,
        help="The predictive factor for the model, which must be one of 8, 16, 32, 64, 128.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
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
        default=None,
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

    training_procedure = NCFCombinedPretrainingProcedure(predictive_factor, batch_size)

    if dataset_name is None:
        logger.info(
            f"Initialized the pretraining procedure for {model_type} with predictive factor {predictive_factor} "
            f"and batch size {batch_size} on random split of the entire dataset."
        )

        dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
        splitter = BalancedSplit(0.95, True)

        train_idx, test_idx = splitter.split(dataset)

        train_dataset = dataset.get_split(train_idx)
        test_dataset = dataset.get_split(test_idx)

        training_procedure.start_training(model_type, train_dataset, test_dataset, NUM_EPOCHS)
    else:
        logger.info(
            f"Initialized the pretraining procedure for {model_type} with predictive factor {predictive_factor} "
            f"and batch size {batch_size} on dataset '{dataset_name}' "
            f"{'without' if val_dataset_name is None else 'with'} validation."
        )

        dataset = RatingsDataset.load(dataset_name)

        val_dataset: Optional[RatingsDataset] = None
        if val_dataset_name is not None:
            val_dataset = RatingsDataset.load(val_dataset_name)
        training_procedure.start_training(model_type, dataset, val_dataset, NUM_EPOCHS)


if __name__ == "__main__":
    main()
