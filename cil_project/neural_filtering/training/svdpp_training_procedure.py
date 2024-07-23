import logging
from typing import Optional

from cil_project.dataset import RatingsDataset
from cil_project.neural_filtering.evaluators import RatingEvaluator
from cil_project.neural_filtering.models import SVDPP
from cil_project.neural_filtering.trainers import AbstractTrainer, SVDPPTrainer
from cil_project.neural_filtering.training.abstract_training_procedure import AbstractTrainingProcedure
from torch import optim
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


class SVDPPTrainingProcedure(AbstractTrainingProcedure):
    """
    Class used to train SVDPP.
    """

    def __init__(self) -> None:
        hyperparameters = {"nr_factors": NR_FACTORS, "lam": LAM, "lam1": LAM1, "lam2": LAM2, "lam3": LAM3}
        super().__init__(hyperparameters, NUM_EPOCHS)

    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        model = SVDPP(self.model_hyperparameters)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)

        evaluator = RatingEvaluator(model, BATCH_SIZE, train_dataset, val_dataset)

        return SVDPPTrainer(model, BATCH_SIZE, optimizer, scheduler, evaluator)


if __name__ == "__main__":
    procedure = SVDPPTrainingProcedure()
    procedure.start_procedure()
