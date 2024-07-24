import argparse
import logging

from cil_project.bayesian_factorization_machines.bfm_models import (
    BayesianFactorizationMachine,
    BayesianFactorizationMachineOP,
)
from cil_project.dataset import BalancedKFold, RatingsDataset, SubmissionDataset
from cil_project.ensembling.utils import write_predictions_to_csv
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train the Bayesian Factorization Machine with the specified rank and number of iterations.
Optionally, grouping, implicit features and ordinal probit can be enabled.

Typical usage:

python ./bfm_training_procedure.py --rank <RANK> --iterations <ITERATIONS> [--dataset <DATASET>]
"""


class BFMTrainingProcedure:
    """
    Class used to train the Bayesian Factorization Machine.
    """

    def __init__(
        self,
        rank: int,
        num_bins: int,
        num_clusters: int,
        iterations: int,
        kfold: int,
        dataset: RatingsDataset,
        grouped: bool = True,
        implicit: bool = True,
        statistical_features: bool = True,
        ordinal_probit: bool = True,
        kmeans: bool = False,
    ) -> None:
        self.kfold = kfold
        self.dataset = dataset
        self.iterations = iterations
        self.grouped = grouped
        self.implicit = implicit
        self.statistical_features = statistical_features
        self.num_bins = num_bins
        self.num_clusters = num_clusters
        self.kmeans = kmeans

        if ordinal_probit:
            self.model = BayesianFactorizationMachineOP(
                rank,
                self.num_bins,
                self.num_clusters,
                self.grouped,
                self.implicit,
                self.statistical_features,
                self.kmeans,
            )
        else:
            self.model = BayesianFactorizationMachine(
                rank,
                self.num_bins,
                self.num_clusters,
                self.grouped,
                self.implicit,
                self.statistical_features,
                self.kmeans,
            )

    def start_training(self) -> None:
        splitter = BalancedKFold(self.kfold, True)
        iterator = splitter.split(self.dataset)

        rmse_values = []

        try:
            for train_indices, test_indices in iterator:
                train_dataset = self.dataset.get_split(train_indices)
                test_dataset = self.dataset.get_split(test_indices)
                rmse_values.append(self.model.train(train_dataset, test_dataset, self.iterations))

            logger.info(f"Mean RMSE: {sum(rmse_values) / len(rmse_values):.4f}")
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

    def final_train(self) -> None:
        self.model.final_train(self.dataset, self.iterations)

    def generate_data_for_stacking(self, train_base_name: str, val_base_name: str, k: int = 10) -> None:
        num_folds = 10
        avg_rmse = 0.0
        for fold in range(k):

            # training and validation split
            train_dataset = RatingsDataset.load(f"{train_base_name}_{fold}")
            val_dataset = RatingsDataset.load(f"{val_base_name}_{fold}")

            # generate predictions for the validation fold
            try:
                avg_rmse += self.model.train(train_dataset, val_dataset, self.iterations)
                inpts = val_dataset.get_inputs()
                preds = self.model.predict(inpts)
                fold_dataset = SubmissionDataset(inpts, preds)
                write_predictions_to_csv(fold_dataset, self.model.get_name(), fold)

            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        avg_rmse /= num_folds
        logger.info(f"Average RMSE over {num_folds} folds: {avg_rmse}")

        # generate predictions for the test set
        try:
            self.final_train()
            self.model.generate_predictions(SUBMISSION_FILE_NAME)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

    # pylint: disable=too-many-locals
    def generate_data_for_blending(self, train_name: str, val_name: str) -> None:
        train_dataset = RatingsDataset.load(train_name)
        val_dataset = RatingsDataset.load(val_name)

        # generate predictions for the validation set
        try:
            self.model.train(train_dataset, val_dataset, self.iterations)
            inpts = val_dataset.get_inputs()
            preds = self.model.predict(inpts)
            fold_dataset = SubmissionDataset(inpts, preds)
            write_predictions_to_csv(fold_dataset, self.model.get_name(), 0)

            # generate predictions for the test set
            self.model.generate_predictions(SUBMISSION_FILE_NAME)

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


# pylint: disable=too-many-locals
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

    parser.add_argument("--kmeans", action="store_true", help="Whether to use kmeans.")

    # blending mode
    parser.add_argument(
        "--blending",
        nargs=2,
        metavar=("TRAINING_DATASET", "VALIDATION_DATASET"),
        help="Run in blending mode with specified training and validation datasets",
    )

    # stacking mode
    parser.add_argument(
        "--stacking",
        nargs=2,
        metavar=("TRAINING_DATASET_BASE", "VALIDATION_DATASET_BASE"),
        help="Run in stacking mode with specified training and validation dataset base names",
    )

    parser.add_argument(
        "--num_bins", type=int, required=False, default=5, help="The number of bins for statistical features."
    )

    parser.add_argument(
        "--num_clusters", type=int, required=False, default=5, help="The number of clusters for kmeans."
    )

    args = parser.parse_args()

    rank: int = args.rank
    iterations: int = args.iterations
    kfold: int = args.kfold
    dataset_name: str = args.dataset
    grouped: bool = args.grouped
    implicit: bool = args.implicit
    statistical_features: bool = args.statistics
    ordinal_probit: bool = args.op
    kmeans: bool = args.kmeans
    num_bins: int = args.num_bins
    num_clusters: int = args.num_clusters

    logger.info(
        f"Initialized the training procedure for the BFM with rank {rank} "
        f"and {iterations} iterations on dataset '{dataset_name}'. "
        f"Grouped: {grouped}, Implicit: {implicit}, Statistics: "
        f"{statistical_features}, Ordinal Probit: {ordinal_probit}, Kmeans: {kmeans}"
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = BFMTrainingProcedure(
        rank,
        num_bins,
        num_clusters,
        iterations,
        kfold,
        dataset,
        grouped,
        implicit,
        statistical_features,
        ordinal_probit,
        kmeans,
    )

    if args.stacking:
        training_dataset_base, validation_dataset_base = args.stacking

        training_procedure.generate_data_for_stacking(training_dataset_base, validation_dataset_base)
    elif args.blending:
        training_dataset, validation_dataset = args.blending

        training_procedure.generate_data_for_blending(training_dataset, validation_dataset)
    else:
        training_procedure.start_training()


if __name__ == "__main__":
    main()
