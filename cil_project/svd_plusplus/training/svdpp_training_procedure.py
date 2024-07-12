import logging

import numpy as np
import torch
from cil_project.dataset import BalancedKFold, BalancedSplit, RatingsDataset, SubmissionDataset
from cil_project.ensembling.utils import write_predictions_to_csv
from cil_project.svd_plusplus.evaluators import SVDPPEvaluator
from cil_project.svd_plusplus.model import SVDPP
from cil_project.svd_plusplus.trainer import SVDPPTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME
from torch import optim
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
This script is used to train svdpp.
Typical usage: python3 cil_project/../svdpp_training_procedure.py
"""

# hyperparameters
LEARNING_RATE = 0.002
DECAY = 0.7
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 5
BATCH_SIZE = 4096
NR_FACTORS = 180
LAM = 0.03
LAM1 = 10.0
LAM2 = 25.0
LAM3 = 10.0


class SVDPPTrainingProcedure:
    """
    Class used to train SVDPP.
    """

    def __init__(self, hyperparameters: dict, batch_size: int, dataset: RatingsDataset) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters

        model = SVDPP(hyperparameters)
        # initialize the trainer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)

        self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

    def start_training(self) -> None:
        splitter = BalancedSplit(0.75, True)

        train_idx, test_idx = splitter.split(self.dataset)

        train_dataset = self.dataset.get_split(train_idx)
        test_dataset = self.dataset.get_split(test_idx)

        try:
            self.trainer.train(train_dataset, test_dataset, NUM_EPOCHS)
            # self.trainer.train(self.dataset, None, NUM_EPOCHS)  # train on whole dataset
        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

    # pylint: disable=too-many-locals
    def start_kfold_training(self, num_folds: int = 10) -> None:
        avg_rmse = 0.0
        kfold = BalancedKFold(num_folds=num_folds, shuffle=True)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.dataset)):
            logger.info(f"Fold {fold+1} started")

            # initialize the trainer
            model = SVDPP(self.hyperparameters)
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
            self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

            # training split
            train_dataset = self.dataset.get_split(train_idx)
            test_dataset = self.dataset.get_split(test_idx)

            try:
                self.trainer.train(train_dataset, test_dataset, NUM_EPOCHS)
                avg_rmse += self.trainer.validation_loss

            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        avg_rmse /= num_folds
        logger.info(f"Average RMSE over {num_folds} folds: {avg_rmse}")

    # pylint: disable=too-many-locals
    def generate_data_for_stacking(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        num_folds = 10
        avg_rmse = 0.0
        for fold in range(10):
            # initialize the trainer
            model = SVDPP(self.hyperparameters)
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
            self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

            # taining and validation split
            train_dataset = RatingsDataset.load(f"stacking_train_{fold}")
            val_dataset = RatingsDataset.load(f"stacking_val_{fold}")

            # generate predictions for the validation fold
            try:
                self.trainer.train(train_dataset, val_dataset, NUM_EPOCHS)
                evaluator = SVDPPEvaluator(model, 64, train_dataset, None)
                inpts = val_dataset.get_inputs()
                preds = evaluator.predict(inpts)
                fold_dataset = SubmissionDataset(inpts, preds)
                write_predictions_to_csv(fold_dataset, evaluator.get_name(), fold)
                avg_rmse += self.trainer.validation_loss

            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        avg_rmse /= num_folds
        logger.info(f"Average RMSE over {num_folds} folds: {avg_rmse}")

        # generate predictions for the test set
        try:
            # initialize the trainer
            model = SVDPP(self.hyperparameters)
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
            self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

            self.trainer.train(self.dataset, None, NUM_EPOCHS)
            evaluator = SVDPPEvaluator(model, 64, self.dataset, None)
            evaluator.generate_predictions(SUBMISSION_FILE_NAME)

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

    # pylint: disable=too-many-locals
    def generate_data_for_blending(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        train_dataset = RatingsDataset.load("blending_train")
        val_dataset = RatingsDataset.load("blending_val")

        # initialize the trainer
        model = SVDPP(self.hyperparameters)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
        self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

        # generate predictions for the validation set
        try:
            self.trainer.train(train_dataset, val_dataset, NUM_EPOCHS)
            evaluator = SVDPPEvaluator(model, 64, train_dataset, None)
            inpts = val_dataset.get_inputs()
            preds = evaluator.predict(inpts)
            fold_dataset = SubmissionDataset(inpts, preds)
            write_predictions_to_csv(fold_dataset, evaluator.get_name(), 0)

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")

        # initialize the trainer
        model = SVDPP(self.hyperparameters)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
        self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

        # generate predictions for the test set
        try:
            self.trainer.train(self.dataset, None, NUM_EPOCHS)
            evaluator = SVDPPEvaluator(model, 64, self.dataset, None)
            evaluator.generate_predictions(SUBMISSION_FILE_NAME)

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user.")


def main() -> None:

    # hyperparameters
    hyperparameters = {"nr_factors": NR_FACTORS, "lam": LAM, "lam1": LAM1, "lam2": LAM2, "lam3": LAM3}

    batch_size = BATCH_SIZE
    dataset_name = FULL_SERIALIZED_DATASET_NAME

    logger.info(
        f"Initialized the training procedure for svdpp with hyperparameters {hyperparameters} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = SVDPPTrainingProcedure(hyperparameters, batch_size, dataset)
    # training_procedure.start_training()
    # training_procedure.start_kfold_training()
    training_procedure.generate_data_for_stacking()
    # training_procedure.generate_data_for_blending()


if __name__ == "__main__":
    main()
