import argparse
import logging

from cil_project.baseline_models.models import ALS, SVP, SVT, Baseline
from cil_project.dataset import RatingsDataset, SubmissionDataset
from cil_project.ensembling.utils import write_predictions_to_csv
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)

logger = logging.getLogger(__name__)

"""
In order to run the training procedures, the following command-line arguments are available:

--model <MODEL_NAME>: Choose SVT, SVP, or ALS as the model for training and prediction.
--blending <TRAINING_DATASET> <VALIDATION_DATASET>: Run in blending mode with
                                                    specified training and validation datasets.
--stacking <TRAINING_DATASET_BASE> <VALIDATION_DATASET_BASE>: run in stacking mode with specified training
                                                              and validation dataset base names.
--verbose: Enable verbosity.

Typical usage:

./python <script_name> [command line arguments]
"""


VERBOSE = False
MODEL_ID = 0

model_name_to_id = {
    "SVT": 0,
    "SVP": 1,
    "ALS": 2,
}

# set hyperparameters for the model here


def get_model(model_id: int) -> Baseline:
    """
    Returns the model with the specified model id.
    :param model_id: the id of the model.
    :return: the model.
    """
    if model_id == 0:
        return SVT(max_iter=60, eta=1.05, tau=37, verbose=VERBOSE)
    if model_id == 1:
        return SVP(max_iter=20, eta=1.3315789473684212, k=4, verbose=VERBOSE)
    if model_id == 2:
        return ALS(max_iter=21, lam=0.2826666666666667, k=3, verbose=VERBOSE)
    raise ValueError("Invalid model id")


def train_model_for_stacking(base_dataset_name: str, base_val_dataset_name: str, k: int = 10) -> None:
    """
    Performs all training/predicting steps for stacking.
    This method essentially performs cross-validation on the provided datasets. Make sure that the dataset names
    are formatted as follows: <base_dataset_name>_<fold_idx> and <base_val_dataset_name>_<fold_idx>.

    :param base_dataset_name: the base name of all training datasets.
    :param base_val_dataset_name: the base name of all validation datasets.
    :param k: number of folds.
    """

    total_rmse = 0.0  # pylint: disable=invalid-name

    for fold in range(k):
        logger.info(f"Perform training for fold {fold+1}/{k}.")

        model = get_model(MODEL_ID)

        train_dataset = RatingsDataset.load(f"{base_dataset_name}_{fold}")
        val_dataset = RatingsDataset.load(f"{base_val_dataset_name}_{fold}")

        # training split
        train_matrix = train_dataset.get_data_matrix()

        # test split
        val_matrix = val_dataset.get_data_matrix()
        val_matrix_mask = val_dataset.get_data_matrix_mask().astype(int)

        # train model
        model.train(train_matrix, val_matrix, val_matrix_mask)

        #  generate predictions for validation fold
        inpts = val_dataset.get_inputs()
        preds = model.predict(inpts)
        fold_dataset = SubmissionDataset(inpts, preds)
        write_predictions_to_csv(fold_dataset, model.get_name(), fold)

        total_rmse += model.rmse

    logger.info(f"Average RMSE over {k} folds: {total_rmse / k}")

    # generate predictions for the test set
    model = get_model(MODEL_ID)
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    logger.info("Train model on full dataset and generate predictions on submission dataset.")
    model.train(dataset.get_data_matrix())
    model.generate_predictions(SUBMISSION_FILE_NAME)
    logger.info("Stacking done.")


def train_model_for_blending(dataset_name: str, test_dataset_name: str) -> None:
    """
    Performs all training/predicting steps for blending.
    That is, the model is trained on the given dataset and predictions are made on the test dataset.
    Then, also all predictions for the submission dataset are made.

    :param dataset_name: the name of the dataset used for blending.
    :param test_dataset_name: the name of the test dataset.
    """

    model = get_model(MODEL_ID)

    train_dataset = RatingsDataset.load(dataset_name)
    val_dataset = RatingsDataset.load(test_dataset_name)

    # training split
    train_matrix = train_dataset.get_data_matrix()

    # test split
    val_matrix = val_dataset.get_data_matrix()
    val_matrix_mask = val_dataset.get_data_matrix_mask().astype(int)

    # train model
    logger.info("Perform training on train set.")
    model.train(train_matrix, val_matrix, val_matrix_mask)

    #  generate predictions for validation fold
    logger.info("Generate predictions for validation set.")
    inpts = val_dataset.get_inputs()
    preds = model.predict(inpts)
    fold_dataset = SubmissionDataset(inpts, preds)
    write_predictions_to_csv(fold_dataset, model.get_name(), 0)

    # generate predictions for the test set
    logger.info("Generate predictions on submission dataset.")
    model.generate_predictions(SUBMISSION_FILE_NAME)
    logger.info("Blending done.")


def main():
    """
    Main method for the training procedure.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Command-line options for different execution modes.")

    parser.add_argument("--model", type=str, choices=model_name_to_id.keys(), help="Choose SVT, SVP or ALS")

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

    parser.add_argument("--verbose", action="store_true", help="Enable verbosity")

    args = parser.parse_args()

    if args.model:
        global MODEL_ID  # pylint: disable=global-statement
        MODEL_ID = model_name_to_id[args.model]

    if args.verbose:
        global VERBOSE  # pylint: disable=global-statement
        VERBOSE = True

    if args.blending and args.stacking:
        logger.info("Attention! Blending and Stacking may write to the same file. Please run them separately.")

    if args.blending:
        logger.info("Perform model training for blending.")
        training_dataset, validation_dataset = args.blending
        train_model_for_blending(training_dataset, validation_dataset)

    if args.stacking:
        logger.info("Perform model training for stacking.")
        training_dataset_base, validation_dataset_base = args.stacking
        train_model_for_stacking(training_dataset_base, validation_dataset_base)


if __name__ == "__main__":
    main()
