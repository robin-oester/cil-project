import argparse
import logging

from cil_project.ensembling import DummyModel, DummyModel2, Ensembler
from cil_project.utils import DATA_PATH, SUBMISSION_FILE_NAME

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
Ensembler class to combine predictions from different predictors.
Typical usage:

./ensembler.py --predictors <predictor1> <predictor2> <predictor3> ...
"""


def run_ensembler() -> None:
    parser = argparse.ArgumentParser(description="Run ensembler with specified prediction predictors.")
    parser.add_argument("--predictors", nargs="+", help="List the predictors you want to use.")
    args = parser.parse_args()

    output_path = Ensembler.combine_predictions(args.predictors, SUBMISSION_FILE_NAME)

    logger.info(f"Successfully combined predictors to file '{output_path}'.")


if __name__ == "__main__":
    # Create sample predictions.
    model1 = DummyModel()
    model2 = DummyModel2()

    model1.generate_predictions(DATA_PATH / SUBMISSION_FILE_NAME)
    model2.generate_predictions(DATA_PATH / SUBMISSION_FILE_NAME)

    run_ensembler()
