import logging
from typing import Optional

from cil_project.dataset import RatingsDataset, TargetNormalization
from cil_project.neural_filtering.evaluators import RatingEvaluator
from cil_project.neural_filtering.models import KANNCF
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
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25
WEIGHT_DECAY = 1e-4
GAMMA = 0.97
EMBEDDING_DIM = 128
HIDDEN_DIM = 32
BATCH_SIZE = 512


class NCFKANProcedure(AbstractTrainingProcedure):
    """
    Class used to perform training of the KAN-based NCF model.
    """

    def __init__(self) -> None:
        hyperparameters = {"embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM}
        super().__init__(hyperparameters, NUM_EPOCHS, TargetNormalization.BY_MOVIE)

    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        model = KANNCF(self.model_hyperparameters)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

        evaluator = RatingEvaluator(model, BATCH_SIZE, train_dataset, val_dataset)

        return RatingTrainer(model, BATCH_SIZE, optimizer, scheduler, evaluator)


if __name__ == "__main__":
    procedure = NCFKANProcedure()
    procedure.start_procedure()
