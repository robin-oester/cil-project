import csv
import pathlib
from datetime import datetime

from dataset.ratings_dataset import RatingsDataset

from .rating_predictor import RatingPredictor


class Ensembler:
    def __init__(self) -> None:
        self.methods: list[RatingPredictor] = []

    def register_method(self, method: RatingPredictor) -> None:
        self.methods.append(method)

    def generate_predictions(self, input_file_path: pathlib.Path, output_folder_path: pathlib.Path) -> None:
        submission_dataset = RatingsDataset(input_file_path)
        predictions_sum: dict[tuple[int, int], float] = {}

        # perform predictions for all methods
        for method in self.methods:
            for (user_idx, movie_idx), _ in submission_dataset:
                prediction = method.predict((user_idx, movie_idx))
                predictions_sum[(user_idx, movie_idx)] = predictions_sum.get((user_idx, movie_idx), 0) + prediction

        # create csv file with predictions
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_file_path = output_folder_path / f"Ensembler_predictions_{current_timestamp}.csv"
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Id", "Prediction"])
            for (user_idx, movie_idx), prediction_sum in predictions_sum.items():
                avg_prediction = prediction_sum / len(self.methods)  # average prediction of all methods
                writer.writerow([f"r{user_idx+1}_c{movie_idx+1}", avg_prediction])
