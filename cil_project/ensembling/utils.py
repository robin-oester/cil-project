import csv
import pathlib
from typing import Optional

from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH


def write_predictions_to_csv(
    submission_dataset: SubmissionDataset, model_name: str, fold_idx: Optional[int] = None
) -> pathlib.Path:
    """
    Write the predictions to a csv file. The predictions are aggregated in the submission dataset.
    Stores them as <model_name>_<fold_idx>.csv if fold_idx is not None, otherwise as <model_name>.csv.

    :param submission_dataset: the submission dataset containing the predictions.
    :param model_name: name of the model.
    :param fold_idx: index of the fold. If None, the model is not part of a k-fold evaluation.
    :return: path to the generated file.
    """

    # create csv file with predictions
    output_file_path = DATA_PATH / (f"{model_name}_{fold_idx}.csv" if fold_idx is not None else f"{model_name}.csv")
    with open(output_file_path, "w", encoding="utf-8", newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Id", "Prediction"])
        for (user_idx, movie_idx), prediction in zip(
            submission_dataset.inputs, submission_dataset.predictions.squeeze()
        ):
            writer.writerow([f"r{user_idx + 1}_c{movie_idx + 1}", prediction])

    return output_file_path
