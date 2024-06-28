import argparse
import logging

from cil_project.dataset import BalancedSplit, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to create a dataset according to your needs.
It will split a given dataset into a training and test set the given split factor.
The dataset can be shuffled before splitting.
Typical usage:

./dataset_creator.py -n <name> --split <split> [--base <base dataset>] [--shuffle]
"""


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

    assert 0.0 <= split <= 1.0, "Split must be between 0 and 1."

    logger.info(
        f"Split dataset '{base_name}' with factor {split} into training and test set "
        f"{'with' if shuffle else 'without'} shuffling."
    )

    # Load the dataset
    dataset = RatingsDataset.load(base_name)
    splitter = BalancedSplit(split, shuffle)
    train_idx, test_idx = splitter.split(dataset)

    train_dataset = dataset.get_split(train_idx)
    test_dataset = dataset.get_split(test_idx)

    train_dataset_name = f"{name}_train"
    test_dataset_name = f"{name}_test"
    train_dataset.store(train_dataset_name)
    test_dataset.store(test_dataset_name)

    logger.info(f"Serialized splits of dataset in train & test with name '{name}'.")


if __name__ == "__main__":
    main()
