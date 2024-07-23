import argparse
import logging
from typing import Optional

from cil_project.dataset import RatingsDataset, SubmissionDataset
from cil_project.ensembling import AvgEnsembler, MetaRegressor
from cil_project.utils import DATA_PATH, SUBMISSION_FILE_NAME
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
Ensembler class to combine predictions from different predictors.
There are 3 types of ensemble implemented:
1. Averaging: Simply average the predictions of the different models.
2. Blending: Blend the predictions of different models using a single validation dataset.
3. Stacking: Stack the predictions of different models in K-fold fashion using multiple validation datasets.
Typical usage:

./ensembler.py --models <model1> <model2> ... [--regressor <regressor>] [--val <val_name>]
"""


# pylint: disable=too-many-branches
def run_ensembler() -> None:
    parser = argparse.ArgumentParser(description="Run ensembler with specified prediction predictors.")
    parser.add_argument("--models", nargs="+", help="List the model names you want to use.")
    parser.add_argument(
        "--regressor",
        type=str,
        choices=["linear", "GradientBoosting"],
        required=False,
        default=None,
        help="The model type, which must be either 'linear', 'GradientBoosting'.",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=False,
        default=None,
        help="The base name of the validation set.",
    )
    args = parser.parse_args()

    model_names = args.models
    regressor_name = args.regressor
    val_name = args.val

    # find the available validation sets
    val_sets: list[RatingsDataset] = []
    if val_name is not None and regressor_name is not None:
        initial_path = DATA_PATH / f"{val_name}.npz"
        if initial_path.is_file():
            # blending
            val_sets.append(RatingsDataset.load(initial_path.stem))
        else:
            # stacking
            idx = 0
            while (path := DATA_PATH / f"{val_name}_{idx}.npz").is_file():
                val_sets.append(RatingsDataset.load(path.stem))
                idx += 1
        logger.info(f"Found {len(val_sets)} validation dataset(s).")

    # choose the correct meta regressor, also add new ones here
    regressor: Optional[RegressorMixin] = None
    if regressor_name == "linear":
        regressor = LinearRegression()
    elif regressor_name == "GradientBoosting":
        regressor = GradientBoostingRegressor()
    elif regressor_name is not None:
        logger.error("Invalid regressor name. Exiting...")
        return

    for model_name in model_names:
        # check whether the model has a prediction for each validation set
        for idx in range(len(val_sets)):
            path = DATA_PATH / f"{model_name}_{idx}.csv"
            if not path.is_file():
                logger.error(f"Missing file {path} for validation set. Exiting...")
                return

        # check whether the model has a test set prediction
        path = DATA_PATH / f"{model_name}.csv"
        if not path.is_file():
            logger.error(f"Missing file {path} for test set submission. Exiting...")
            return

    if regressor is not None:
        if len(val_sets) == 1:
            logger.info(f"Perform blending of {model_names}.")
        elif len(val_sets) > 1:
            logger.info(f"Perform stacking of {model_names}.")
        ensembler = MetaRegressor(regressor, model_names, val_sets)
    else:
        logger.info(f"Perform averaging of {model_names}.")
        ensembler = AvgEnsembler(model_names)

    submission = SubmissionDataset.from_file(DATA_PATH / SUBMISSION_FILE_NAME)
    final_path = ensembler.predict(submission)

    logger.info(f"Stored final predictions to '{final_path.name}'.")


if __name__ == "__main__":
    run_ensembler()
