import logging
import os
import csv

import numpy as np
import torch
from cil_project.dataset import BalancedKFold, BalancedSplit, RatingsDataset, SubmissionDataset
from cil_project.svd_plusplus.evaluators import SVDPPEvaluator
from cil_project.svd_plusplus.model import SVDPP
from cil_project.svd_plusplus.trainer import SVDPPTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, DATA_PATH, SUBMISSION_FILE_NAME
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


def store_to_csv(inputs: np.ndarray, predictions: np.ndarray, file_name: str) -> None:
    """
    Store the predictions to a csv file.

    :param inputs: The inputs for which the predictions were made. Expected shape (N, 2).
    :param predictions: The predictions. Expected shape (N,1).
    :param file_name: The file name of the csv file.
    """
    file_path = DATA_PATH / f"{file_name}.csv"
    
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["Id", "Prediction"])
        
        for i in range(inputs.shape[0]):
            row_id = f"r{inputs[i, 0]+1}_c{inputs[i, 1]+1}"
            writer.writerow([row_id, predictions[i, 0]])


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
            # self.trainer.train(self.dataset, None, NUM_EPOCHS) # train on whole dataset
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

            try:
                self.trainer.train(train_dataset, val_dataset, NUM_EPOCHS)
                evaluator = SVDPPEvaluator(model, 64, train_dataset, None)
                preds = evaluator.predict(val_dataset.get_inputs())
                store_to_csv(val_dataset.get_inputs(), preds, f"{evaluator.get_name()}_{fold}")
                avg_rmse += self.trainer.validation_loss

            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        avg_rmse /= num_folds
        logger.info(f"Average RMSE over {num_folds} folds: {avg_rmse}")
        try:
            # initialize the trainer
            model = SVDPP(self.hyperparameters)
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
            self.trainer = SVDPPTrainer(model, self.batch_size, optimizer, scheduler)

            self.trainer.train(self.dataset, None, NUM_EPOCHS)
            evaluator = SVDPPEvaluator(model, 64, self.dataset, None)
            inpts = SubmissionDataset(DATA_PATH / SUBMISSION_FILE_NAME).inputs
            preds = evaluator.predict(inpts)
            store_to_csv(inpts, preds, f"{evaluator.get_name()}")

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
    training_procedure.generate_data_for_stacking()
    # training_procedure.start_training()
    # training_procedure.start_kfold_training()


if __name__ == "__main__":
    main()
