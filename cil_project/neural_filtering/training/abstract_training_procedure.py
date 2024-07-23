import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from cil_project.dataset import BalancedSplit, RatingsDataset, SubmissionDataset, TargetNormalization
from cil_project.ensembling.utils import write_predictions_to_csv
from cil_project.neural_filtering.trainers import AbstractTrainer
from cil_project.utils import DATA_PATH, FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME

logger = logging.getLogger(__name__)

"""
In order to run the training procedures, the following command-line arguments are available:

--test: run in test mode
--train <DATASET> <VALIDATION_DATASET>: run in training mode with specified train dataset
                                        and optional validation dataset
--checkpoint_granularity <num>: optional int specifying checkpoint granularity for test/train mode
--blending <TRAINING_DATASET> <VALIDATION_DATASET>: run in blending mode with specified training and validation datasets
--stacking <TRAINING_DATASET_BASE> <VALIDATION_DATASET_BASE>: run in stacking mode with specified training
                                                              and validation dataset base names

All options can be combined arbitrarily. Typical usage is then:

./python <script_name> [command line arguments]
"""


class AbstractTrainingProcedure(ABC):
    """
    Abstract class for all training procedures.
    """

    def __init__(
        self,
        model_hyperparameters: dict[str, Any],
        num_training_epochs: int,
        normalization: Optional[TargetNormalization] = None,
    ) -> None:
        """
        Initialize the training procedure with the specified hyperparameters and number of training epochs.

        :param model_hyperparameters: hyperparameters of the model.
        :param num_training_epochs: number of epochs to train the model.
        :param normalization: (optional) normalization type to apply to the dataset.
        """

        self.model_hyperparameters = model_hyperparameters
        self.num_training_epochs = num_training_epochs
        self.normalization = normalization

    @abstractmethod
    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        """
        Get a new trainer for the given training and optional validation dataset.

        :param train_dataset: the dataset used for training.
        :param val_dataset: (optional) validation dataset.
        :return: a new trainer instance.
        """
        raise NotImplementedError()

    def test_model(self, checkpoint_granularity: Optional[int] = None) -> None:
        """
        Perform testing of the model for, e.g., hyperparameter tuning.
        Creates a random split of the entire dataset and trains the model on the training set.

        :param checkpoint_granularity: number of epochs before storing a new checkpoint.
        """

        dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

        splitter = BalancedSplit(0.95, True)
        train_idx, val_idx = splitter.split(dataset)

        train_dataset = dataset.get_split(train_idx)
        val_dataset = dataset.get_split(val_idx)

        # normalize the training dataset
        if self.normalization is not None:
            train_dataset.normalize(self.normalization)

        trainer = self.get_trainer(train_dataset, val_dataset)

        try:
            trainer.train(dataset, self.num_training_epochs, checkpoint_granularity)
        except KeyboardInterrupt:
            logger.info("Testing interrupted by the user.")

    def train_model(
        self, dataset_name: str, val_dataset_name: Optional[str], checkpoint_granularity: Optional[int] = None
    ) -> None:
        """
        Train the model on a given dataset and optional validation dataset.

        :param dataset_name: the name of the training set.
        :param val_dataset_name: (optional) name of the validation set.
        :param checkpoint_granularity: number of epochs before storing a new checkpoint.
        """

        train_dataset = RatingsDataset.load(dataset_name)

        val_dataset = None
        if val_dataset_name is not None:
            val_dataset = RatingsDataset.load(val_dataset_name)

        # normalize the training dataset
        if self.normalization is not None:
            train_dataset.normalize(self.normalization)

        trainer = self.get_trainer(train_dataset, val_dataset)

        try:
            trainer.train(train_dataset, self.num_training_epochs, checkpoint_granularity)
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

    def train_model_for_blending(self, dataset_name: str, test_dataset_name: str) -> None:
        """
        Performs all training/predicting steps for blending.
        That is, the model is trained on the given dataset and predictions are made on the test dataset.
        Then, also all predictions for the submission dataset are made.

        :param dataset_name: the name of the dataset used for blending.
        :param test_dataset_name: the name of the test dataset.
        """

        train_dataset = RatingsDataset.load(dataset_name)
        test_dataset = RatingsDataset.load(test_dataset_name)

        # normalize the training dataset
        if self.normalization is not None:
            train_dataset.normalize(self.normalization)

        trainer = self.get_trainer(train_dataset, test_dataset)

        results, _ = AbstractTrainingProcedure._perform_prediction(
            trainer, train_dataset, self.num_training_epochs, test_dataset.get_inputs()
        )

        ensemble_result = SubmissionDataset(test_dataset.get_inputs(), results)
        ensemble_path = write_predictions_to_csv(ensemble_result, trainer.model.__class__.__name__, 0)

        logger.info(f"Stored predictions for ensemble to '{ensemble_path.name}'.")

        # generate predictions for the submission set
        submission = SubmissionDataset.from_file(DATA_PATH / SUBMISSION_FILE_NAME)
        submission.predictions = trainer.evaluator.predict(submission.inputs)

        submission_path = write_predictions_to_csv(submission, trainer.model.__class__.__name__)

        logger.info(f"Stored predictions for submission dataset to '{submission_path.name}'")

    def train_model_for_stacking(self, base_dataset_name: str, base_val_dataset_name: str, k: int = 10) -> None:
        """
        Performs all training/predicting steps for stacking.
        This method essentially performs cross-validation on the provided datasets. Make sure that the dataset names
        are formatted as follows: <base_dataset_name>_<fold_idx> and <base_val_dataset_name>_<fold_idx>.

        :param base_dataset_name: the base name of all training datasets.
        :param base_val_dataset_name: the base name of all validation datasets.
        :param k: number of folds.
        """

        total_rmse = 0.0
        for i in range(k):
            logger.info(f"Perform training for fold {i+1}/{k}.")
            train_dataset_name = f"{base_dataset_name}_{i}"
            val_dataset_name = f"{base_val_dataset_name}_{i}"

            train_dataset = RatingsDataset.load(train_dataset_name)
            val_dataset = RatingsDataset.load(val_dataset_name)

            # normalize the training dataset
            if self.normalization is not None:
                train_dataset.normalize(self.normalization)

            trainer = self.get_trainer(train_dataset, val_dataset)

            results, rmse = AbstractTrainingProcedure._perform_prediction(
                trainer, train_dataset, self.num_training_epochs, val_dataset.get_inputs()
            )
            total_rmse += rmse  # type: ignore

            ensemble_result = SubmissionDataset(val_dataset.get_inputs(), results)
            ensemble_path = write_predictions_to_csv(ensemble_result, trainer.model.__class__.__name__, i)

            logger.info(f"Stored predictions for ensemble to '{ensemble_path.name}'.")

        logger.info(f"Finished training of all {k} folds.")
        logger.info(f"Average RMSE: {(total_rmse / k):.4f}")

    def predict_submission(self) -> None:
        """
        Trains the model on the entire dataset and performs predictions on the submission dataset.
        """

        train_dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

        # normalize the training dataset
        if self.normalization is not None:
            train_dataset.normalize(self.normalization)

        trainer = self.get_trainer(train_dataset)

        submission_dataset = SubmissionDataset.from_file(DATA_PATH / SUBMISSION_FILE_NAME)
        results, _ = AbstractTrainingProcedure._perform_prediction(
            trainer, train_dataset, self.num_training_epochs, submission_dataset.inputs
        )
        submission_dataset.predictions = results
        submission_path = write_predictions_to_csv(submission_dataset, trainer.model.__class__.__name__)

        logger.info(f"Stored predictions of the submission dataset to '{submission_path.name}'.")

    @staticmethod
    def _perform_prediction(
        trainer: AbstractTrainer, dataset: RatingsDataset, num_epochs: int, test_inputs: np.ndarray
    ) -> tuple[np.ndarray, Optional[float]]:
        assert trainer.evaluator is not None, "Trainer must have evaluator to perform prediction."

        last_val = trainer.train(dataset, num_epochs)
        results = trainer.evaluator.predict(test_inputs)

        return results, last_val

    def start_procedure(self) -> None:
        """
        Start the procedure.
        """

        # Initialize the argument parser
        parser = argparse.ArgumentParser(description="Command-line options for different execution modes.")

        # testing mode
        parser.add_argument("--test", action="store_true", help="Run in test mode")

        # training mode
        parser.add_argument(
            "--train",
            nargs="*",
            metavar="DATASET",
            help="Run in training mode with specified train dataset and optional validation dataset",
        )

        parser.add_argument(
            "--checkpoint_granularity",
            type=int,
            default=None,
            help="Optional int specifying checkpoint granularity for test/train mode",
        )

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

        args = parser.parse_args()

        # Execute the code based on the arguments
        if args.test:
            checkpoint_granularity = args.checkpoint_granularity
            if checkpoint_granularity is None:
                logger.info("Perform testing of the model without checkpointing.")
            else:
                logger.info(f"Perform testing of the model with checkpoint granularity {args.checkpoint_granularity}.")
            self.test_model(checkpoint_granularity)

        if args.train:
            train_dataset_name = args.train[0] if len(args.train) > 0 else FULL_SERIALIZED_DATASET_NAME
            validation_dataset_name = args.train[1] if len(args.train) > 1 else None

            checkpoint_granularity = args.checkpoint_granularity
            if checkpoint_granularity is None:
                logger.info("Perform training of the model without checkpointing.")
            else:
                logger.info(f"Perform training of the model with checkpoint granularity {args.checkpoint_granularity}.")

            self.train_model(train_dataset_name, validation_dataset_name, checkpoint_granularity)

        if args.blending and args.stacking:
            logger.info("Attention! Blending and Stacking may write to the same file. Please run them separately.")

        if args.blending:
            logger.info("Perform model training for blending.")
            training_dataset, validation_dataset = args.blending
            self.train_model_for_blending(training_dataset, validation_dataset)

        if args.stacking:
            logger.info("Perform model training for stacking.")
            training_dataset_base, validation_dataset_base = args.stacking
            self.train_model_for_stacking(training_dataset_base, validation_dataset_base)

            logger.info("Perform full training for predicting the submission dataset.")
            self.predict_submission()
