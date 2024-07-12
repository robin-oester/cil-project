import logging
from typing import Optional

from cil_project.dataset import RatingsDataset, TargetNormalization
from cil_project.neural_filtering.evaluators import ReconstructionEvaluator
from cil_project.neural_filtering.models import Autoencoder
from cil_project.neural_filtering.trainers import AbstractTrainer, ReconstructionTrainer
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
NUM_EPOCHS = 300
GAMMA = 0.99
WEIGHT_DECAY = 1e-4
ENCODING_SIZE = 32
BATCH_SIZE = 64
DROPOUT = 0.5


class AutoencoderProcedure(AbstractTrainingProcedure):
    """
    Class used to train the autoencoder.
    """

    def __init__(self) -> None:
        hyperparameters = {"encoding_size": ENCODING_SIZE, "p_dropout": DROPOUT}
        super().__init__(hyperparameters, NUM_EPOCHS, TargetNormalization.BY_MOVIE)

    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        model = Autoencoder(self.model_hyperparameters)

        optimizer = Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

        evaluator = ReconstructionEvaluator(model, BATCH_SIZE, train_dataset, val_dataset)

        return ReconstructionTrainer(model, BATCH_SIZE, optimizer, scheduler, evaluator)


if __name__ == "__main__":
    procedure = AutoencoderProcedure()
    procedure.start_procedure()
