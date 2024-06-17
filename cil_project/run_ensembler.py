import argparse
import pathlib

from ensembling.dummy_model import DummyModel, DummyModel2
from ensembling.ensembler import Ensembler

#  Example: python3 cil_project/run_ensembler.py --predictors DummyModel DummyModel2 --use_csv
if __name__ == "__main__":

    # Register predictors here
    predictor_registry: dict[str, type] = {"DummyModel": DummyModel, "DummyModel2": DummyModel2}

    #  Parse arguments
    parser = argparse.ArgumentParser(description="Run ensembler with specified prediction predictors.")
    parser.add_argument("--predictors", nargs="+", help="List the predictors you want to use.")
    parser.add_argument("--use_csv", action="store_true", help="Use the generate_predictions_from_csv method")
    args = parser.parse_args()

    # Initialize ensembler
    ensembler = Ensembler()

    # Register predictors
    for predictor in args.predictors:
        if predictor in predictor_registry:
            ensembler.register_predictor(predictor_registry[predictor]())
        else:
            raise ValueError(f"Predictor {predictor} not found in registry.")

    # Generate predictions
    input_file_path = pathlib.Path("./data/sampleSubmission.csv")
    output_folder_path = pathlib.Path("./data/outputs")

    if args.use_csv:  # In order to use the generate_predictions_from_csv method
        # the predictors must have generated predictions before
        # and the files must be in the output folder
        ensembler.generate_predictions_from_csv(output_folder_path)
    else:
        ensembler.generate_predictions(input_file_path, output_folder_path)

    # Example of how to generate predictions for a single predictor
    predictor_registry["DummyModel"]().generate_predictions(input_file_path, output_folder_path)
    predictor_registry["DummyModel2"]().generate_predictions(input_file_path, output_folder_path)
