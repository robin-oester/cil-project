import logging
from typing import Optional

from cil_project.dataset import RatingsDataset, TargetNormalization
from cil_project.neural_filtering.evaluators import RatingEvaluator
from cil_project.neural_filtering.models import NCFMLPModel
from cil_project.neural_filtering.trainers import AbstractTrainer, RatingTrainer
from cil_project.neural_filtering.training.abstract_training_procedure import AbstractTrainingProcedure
from torch import optim
from torch.optim import Adam

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


# learning constants
NUM_EPOCHS = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA = 0.85
PREDICTIVE_FACTOR = 128
BATCH_SIZE = 512


class NCFMLPTrainingProcedure(AbstractTrainingProcedure):
    """
    Procedure for training the MLP part of the combined NCF model.
    """

    def __init__(self) -> None:
        hyperparameters = {"predictive_factor": PREDICTIVE_FACTOR}
        super().__init__(hyperparameters, NUM_EPOCHS, TargetNormalization.BY_MOVIE)

    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        model = NCFMLPModel(self.model_hyperparameters)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

        evaluator = RatingEvaluator(model, BATCH_SIZE, train_dataset, val_dataset)

        return RatingTrainer(model, BATCH_SIZE, optimizer, scheduler, evaluator)


if __name__ == "__main__":
    procedure = NCFMLPTrainingProcedure()
    procedure.start_procedure()
