import logging
import os
from cil_project.dataset import BalancedKFold, BalancedSplit, RatingsDataset
from cil_project.svd_plusplus.model import SVDPP
from cil_project.svd_plusplus.trainer import SVDPPTrainer
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from cil_project.svd_plusplus.evaluators import SVDPPEvaluator
from torch import optim
from torch.optim import Adam
import torch
import numpy as np

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


STORE_KFOLD = False
if STORE_KFOLD:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)



LEARNING_RATE = 0.002
DECAY = 0.7
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 5


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
                if STORE_KFOLD:
                    print("length of test dataset: ", len(test_dataset))
                    evaluator = SVDPPEvaluator(model, 64, train_dataset, None)
                    preds = evaluator.predict(test_dataset.get_inputs())
                    file_path = os.path.join(os.path.dirname(__file__), "svdpp_kfold_results.txt")
                    with open(file_path, 'ab') as f:
                        np.savetxt(f, preds, fmt='%f')

                avg_rmse += self.trainer.validation_loss

            except KeyboardInterrupt:
                logger.info("Training interrupted by the user.")

        avg_rmse /= num_folds
        logger.info(f"Average RMSE over {num_folds} folds: {avg_rmse}")


def main() -> None:

    # hyperparameters
    hyperparameters = {"nr_factors": 180, "lam": 0.03, "lam1": 10.0, "lam2": 25.0, "lam3": 10.0}

    batch_size = 4096  # works fine
    dataset_name = FULL_SERIALIZED_DATASET_NAME

    logger.info(
        f"Initialized the training procedure for svdpp with hyperparameters {hyperparameters} "
        f"and batch size {batch_size} on dataset '{dataset_name}'."
    )

    dataset = RatingsDataset.load(dataset_name)
    training_procedure = SVDPPTrainingProcedure(hyperparameters, batch_size, dataset)
    # training_procedure.start_training()
    training_procedure.start_kfold_training()


if __name__ == "__main__":
    main()
