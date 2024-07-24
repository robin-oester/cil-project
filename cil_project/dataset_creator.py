import argparse
import logging

from cil_project.dataset import BalancedKFold, BalancedSplit, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to create a dataset according to your needs.
It either creates a simple train/validation set split or a k-fold split of a given dataset.
The factor needs to be in [0, 1] for a simple split and an integer greater than 1 for k-folds.
The dataset can optionally be shuffled before splitting.
Typical usage:

python ./dataset_creator.py -n <name> --split <split> [--base <base dataset>] [--shuffle]
"""


def create_split(dataset: RatingsDataset, name: str, split: float, shuffle: bool) -> None:
    logger.info(
        f"Split dataset into training and validation set with factor {split} "
        f"{'with' if shuffle else 'without'} shuffling."
    )

    # Load the dataset
    splitter = BalancedSplit(split, shuffle)
    train_idx, val_idx = splitter.split(dataset)

    train_dataset = dataset.get_split(train_idx)
    val_dataset = dataset.get_split(val_idx)

    train_dataset_name = f"{name}_train"
    val_dataset_name = f"{name}_val"
    train_dataset.store(train_dataset_name)
    val_dataset.store(val_dataset_name)

    logger.info(f"Serialized splits of dataset in train & validation with name '{name}'.")


def create_k_folds(dataset: RatingsDataset, name: str, num_folds: int, shuffle: bool) -> None:
    logger.info(f"Split dataset into {num_folds} folds " f"{'with' if shuffle else 'without'} shuffling.")

    # Load the dataset
    splitter = BalancedKFold(num_folds, shuffle)

    for idx, (train_idx, val_idx) in enumerate(splitter.split(dataset)):
        train_dataset = dataset.get_split(train_idx)
        val_dataset = dataset.get_split(val_idx)

        train_dataset_name = f"{name}_train_{idx}"
        val_dataset_name = f"{name}_val_{idx}"

        train_dataset.store(train_dataset_name)
        val_dataset.store(val_dataset_name)

    logger.info(f"Serialized folds of dataset in train & validation with name '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dataset according to your needs.")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="The name of the dataset.",
    )

    parser.add_argument(
        "--split",
        type=float,
        required=True,
        help="Percentage to be in the training set.",
    )

    parser.add_argument(
        "--base",
        type=str,
        choices=RatingsDataset.get_available_dataset_names(),
        required=False,
        default=FULL_SERIALIZED_DATASET_NAME,
        help="The name of the base dataset.",
    )

    # Add the shuffle argument as set flag
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset.",
    )

    args = parser.parse_args()

    name: str = args.name
    base_name: str = args.base
    split: float = args.split
    shuffle: bool = args.shuffle

    dataset = RatingsDataset.load(base_name)
    if 0.0 <= split <= 1.0:
        create_split(dataset, name, split, shuffle)
    elif split.is_integer() and split > 1.0:
        create_k_folds(dataset, name, int(split), shuffle)
    else:
        raise ValueError("Split must be between 0 and 1 (simple split) or an integer greater than 1 (folds).")


if __name__ == "__main__":
    main()
