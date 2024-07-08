import pathlib

import numpy as np
from cil_project.dataset import RatingsDataset, SubmissionDataset
from cil_project.utils import DATA_PATH, MAX_RATING, MIN_RATING
from sklearn.base import RegressorMixin

from .abstract_ensembler import AbstractEnsembler
from .utils import write_predictions_to_csv


class MetaRegressor(AbstractEnsembler):
    """
    Combines predictions from different models using a regressor.
    This class enables two types of ensemble:
    - Stacking: Use k-fold cross validation results to train the regressor.
    - Blending: Use a hold-out validation set to train the regressor.
    The subsequent sections describe them in more detail and show how to use the class.

    Stacking:
    1. Use the dataset creator to create k folds of the training set.
       E.g. for 10 folds: ./dataset_creator.py -n stacking --split 10 --shuffle
    2. Train the models on each training set and predict the validation sets.
       Name these files "<model_name>_<fold_idx>.csv".
    3. Train the models on the entire training set to predict the test set.
       Name these files "<model_name>.csv".
    4. Initialize the MetaRegressor using a base model, the model names and the validation sets of each fold.
    5. Call the predict method with the model names and the test set.

    Blending:
    1. Use the dataset creator to create a training and validation set.
       E.g. for a 0.9/0.1 split: ./dataset_creator.py -n blending --split 0.9 --shuffle
    2. Train the models on the training set and predict the validation set.
       Name these files "<model_name>_0.csv".
    3. Use the same models to predict the test set.
       Name these files "<model_name>.csv".
    4. Initialize the MetaRegressor using a base model, the model names and the validation set.
    5. Call the predict method with the model names and the test set.
    """

    def __init__(self, base_model: RegressorMixin, model_names: list[str], val_sets: list[RatingsDataset]) -> None:
        """
        Initialize the MetaRegressor for a given base model.
        Expects the models to have stored the predictions in files "<model_name>_<i>.csv" for the i-th fold.
        If one wants to use blending, it is sufficient to have the predictions stored in "<model_name>_0.csv".

        :param base_model: a base model like `LinearRegression`, `GradientBoostingRegressor`, etc.
        :param model_names: the name of the models to be used for the ensemble.
        :param val_sets: the validation sets for each fold (just one in case of blending).
        """

        super().__init__(model_names)

        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for idx, dataset in enumerate(val_sets):
            assert not dataset.is_normalized(), "Dataset must not be normalized."

            paths = [DATA_PATH / f"{name}_{idx}.csv" for name in model_names]
            fold_preds = MetaRegressor.load_predictions(paths, dataset.get_inputs())
            predictions.append(fold_preds)
            targets.append(dataset.get_targets())

        base_model.fit(np.concatenate(predictions, axis=0).squeeze(), np.concatenate(targets, axis=0).squeeze())

        self.model = base_model
        self.model_names = model_names

    def predict(self, submission_dataset: SubmissionDataset) -> pathlib.Path:
        """
        Predict the submission dataset using the trained regressor.
        Expects the individual models to have stored the predictions in a file `<model_name>.csv`.

        :param submission_dataset: the submission dataset to predict.
        """

        paths = [DATA_PATH / f"{name}.csv" for name in self.model_names]

        model_predictions = MetaRegressor.load_predictions(paths, submission_dataset.inputs)

        submission_dataset.predictions = self.model.predict(model_predictions)
        submission_dataset.predictions = np.expand_dims(
            np.clip(submission_dataset.predictions, MIN_RATING, MAX_RATING), axis=1
        )

        return write_predictions_to_csv(submission_dataset, self.__class__.__name__)
