import argparse
import pathlib

from ensembling.dummy_model import DummyModel, DummyModel2
from ensembling.ensembler import Ensembler
from ensembling.rating_predictor import RatingPredictor

#  Example: python3 cil_project/run_ensembler.py --methods DummyModel DummyModel2
if __name__ == "__main__":

    # Register methods here
    method_registry: dict[str, RatingPredictor] = {"DummyModel": DummyModel(), "DummyModel2": DummyModel2()}

    #  Parse arguments
    parser = argparse.ArgumentParser(description="Run ensembler with specified prediction methods.")
    parser.add_argument("--methods", nargs="+", help="List the methods you want to use.")
    args = parser.parse_args()

    # Initialize ensembler
    ensembler = Ensembler()

    # Register methods
    for method in args.methods:
        if method in method_registry:
            ensembler.register_method(method_registry[method])
        else:
            raise ValueError(f"Method {method} not found in registry.")

    # Generate predictions
    input_file_path = pathlib.Path("./data/sampleSubmission.csv")
    output_folder_path = pathlib.Path("./outputs")
    ensembler.generate_predictions(input_file_path, output_folder_path)
