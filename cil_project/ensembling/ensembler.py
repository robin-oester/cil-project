import csv
import pathlib
import time

from dataset.submission_dataset import SubmissionDataset

from .rating_predictor import RatingPredictor


class Ensembler:
    def __init__(self) -> None:
        self.predictors: list[RatingPredictor] = []

    def register_predictor(self, predictor: RatingPredictor) -> None:
        self.predictors.append(predictor)

    def generate_predictions(self, input_file_path: pathlib.Path, output_folder_path: pathlib.Path) -> None:
        submission_dataset = SubmissionDataset(input_file_path)
        predictions: dict[tuple[int, int], float] = {}

        # perform predictions for all predictors
        for predictor in self.predictors:
            for (user_idx, movie_idx), _ in submission_dataset:
                prediction = predictor.predict((user_idx, movie_idx))
                predictions[(user_idx, movie_idx)] = predictions.get((user_idx, movie_idx), 0) + prediction

        for (user_idx, movie_idx), prediction_sum in predictions.items():
            avg_prediction = prediction_sum / len(self.predictors)
            predictions[(user_idx, movie_idx)] = avg_prediction

        self.write_predictions_to_csv(predictions, output_folder_path)

    def generate_predictions_from_csv(self, output_folder_path: pathlib.Path) -> None:
        predictions: dict[tuple[int, int], float] = {}

        # perform predictions for all predictors
        for predictor in self.predictors:
            predictor_path = self.find_newest_predictor_file(output_folder_path, predictor.name)
            predictor_dataset = SubmissionDataset(predictor_path)
            for (user_idx, movie_idx), prediction in predictor_dataset:
                predictions[(user_idx, movie_idx)] = predictions.get((user_idx, movie_idx), 0) + prediction

        for (user_idx, movie_idx), prediction_sum in predictions.items():
            avg_prediction = prediction_sum / len(self.predictors)
            predictions[(user_idx, movie_idx)] = avg_prediction

        self.write_predictions_to_csv(predictions, output_folder_path)

    def write_predictions_to_csv(
        self, predictions: dict[tuple[int, int], float], output_folder_path: pathlib.Path
    ) -> None:
        # create csv file with predictions
        millis = int(time.time())
        output_file_path = output_folder_path / f"Ensembler_{millis}.csv"
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Id", "Prediction"])
            for (user_idx, movie_idx), prediction in predictions.items():
                writer.writerow([f"r{user_idx+1}_c{movie_idx+1}", prediction])

    def find_newest_predictor_file(self, output_folder_path: pathlib.Path, predictor_name: str) -> pathlib.Path:
        # Pattern to match "predictorName_TIMESTAMP.csv"
        pattern = f"{predictor_name}_*.csv"

        # List all matching files
        files = list(output_folder_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found in {output_folder_path} matching pattern {pattern}.")

        # Function to extract timestamp from filename
        def extract_timestamp(file_path: pathlib.Path) -> int:
            # Extract the part of the filename between the last underscore and ".csv"
            filename = file_path.name
            start = filename.rfind("_") + 1
            end = filename.rfind(".csv")
            timestamp = filename[start:end]
            return int(timestamp) if timestamp.isdigit() else 0

        # Find the file with the largest timestamp
        newest_file = max(files, key=extract_timestamp, default=None)
        if newest_file is None:
            raise ValueError(f"No valid file found in {output_folder_path} matching pattern {pattern}.")

        # Return the newest file
        return newest_file
