import argparse
import logging

from cil_project.bayesian_factorization_machines.models.bayesian_factorization_machine import (
    BayesianFactorizationMachine,
)
from cil_project.bayesian_factorization_machines.models.bayesian_factorization_machine_op import (
    BayesianFactorizationMachineOP,
)
from cil_project.dataset import BalancedKFold, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train the Bayesian Factorization Machine with the specified rank and number of iterations. \
    Optionally, grouping, implicit features and ordinal probit can be enabled.

./bfm_training_procedure.py --rank <rank> --iterations <iterations> [--dataset <dataset>]
"""


class BFMTrainingProcedure:
    """
    Class used to train the Bayesian Factorization Machine.
    """

    def __init__(
        self,
        rank: int,
        iterations: int,
        kfold: int,
        dataset: RatingsDataset,
        grouped: bool = True,
        implicit: bool = True,
        statistical_features: bool = True,
        ordinal_probit: bool = True,
    ) -> None:
        self.kfold = kfold
        self.dataset = dataset
        self.iterations = iterations
        self.grouped = grouped
        self.implicit = implicit
        self.statistical_features = statistical_features

        if ordinal_probit:
            self.model = BayesianFactorizationMachineOP(rank, self.grouped, self.implicit, self.statistical_features)
        else:
            self.model = BayesianFactorizationMachine(rank, self.grouped, self.implicit, self.statistical_features)

    def start_training(self) -> None:
        splitter = BalancedKFold(self.kfold, True)
        iterator = splitter.split(self.dataset)

        rmse_values = []

        for train_indices, test_indices in iterator:
            train_dataset = self.dataset.get_split(train_indices)
            test_dataset = self.dataset.get_split(test_indices)
            try:
                rmse_values.append(self.model.train(train_dataset, test_dataset, self.iterations))
            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        print("Mean RMSE: ", sum(rmse_values) / len(rmse_values))

    def final_train(self) -> None:
        self.model.final_train(self.dataset, self.iterations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Bayesian Factorization Machine with the specified rank and number of iterations. \
            Optionally, grouping, implicit features and ordinal probit can be enabled."
    )

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="The rank of the BFM. Must be a positive integer.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="The number of iterations for the BFM. Must be a positive integer.",
    )

    parser.add_argument(
        "--kfold",
        type=int,
        required=False,
        default=5,
        help="The number of folds for the kfold cross-validation. Must be a positive integer.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=RatingsDataset.get_available_dataset_names(),
        required=False,
        default=FULL_SERIALIZED_DATASET_NAME,
        help="The name of the dataset.",
    )
    parser.add_argument("--grouped", action="store_true", help="Whether to use grouped features.")

    parser.add_argument("--implicit", action="store_true", help="Whether to use implicit features.")

    parser.add_argument("--statistics", action="store_true", help="Whether to use statistical features.")

    parser.add_argument("--op", action="store_true", help="Whether to use ordinal probit.")

    args = parser.parse_args()

    rank: int = args.rank
    iterations: int = args.iterations
    kfold: int = args.kfold
    dataset_name: str = args.dataset
    grouped: bool = args.grouped
    implicit: bool = args.implicit
    statistical_features: bool = args.statistics
    ordinal_probit: bool = args.op

    logger.info(
        f"Initialized the training procedure for the BFM with rank {rank} "
        f"and {iterations} iterations on dataset '{dataset_name}'."
        f"Grouped: {grouped}, Implicit: {implicit}, Statistics: {
            statistical_features}, Ordinal Probit: {ordinal_probit}"
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = BFMTrainingProcedure(
        rank, iterations, kfold, dataset, grouped, implicit, statistical_features, ordinal_probit
    )
    training_procedure.start_training()


if __name__ == "__main__":
    main()
