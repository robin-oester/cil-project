import logging
from typing import Optional

import torch
from cil_project.dataset import RatingsDataset, TargetNormalization
from cil_project.neural_filtering.evaluators import RatingEvaluator
from cil_project.neural_filtering.models import NCFCombined
from cil_project.neural_filtering.trainers import AbstractTrainer, RatingTrainer
from cil_project.neural_filtering.training.abstract_training_procedure import AbstractTrainingProcedure
from cil_project.utils import CHECKPOINT_PATH
from torch.optim import SGD

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

# learning constants
NUM_EPOCHS = 1
ALPHA = 0.5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 2e-3
PREDICTIVE_FACTOR = 128
BATCH_SIZE = 1024

# Pretrained model checkpoints
GMF_PRETRAINED_NAME = "NCFGMF_blending.pkl"
MLP_PRETRAINED_NAME = "NCFMLP_blending.pkl"


class NCFCombinedTrainingProcedure(AbstractTrainingProcedure):
    """
    Procedure used to train the NCF model. Make sure that the pretrained model checkpoints are available
    and are trained on the correct dataset.
    """

    def __init__(self) -> None:
        hyperparameters = {"predictive_factor": PREDICTIVE_FACTOR, "alpha": ALPHA}
        super().__init__(hyperparameters, NUM_EPOCHS, TargetNormalization.BY_MOVIE)

    def get_trainer(
        self, train_dataset: RatingsDataset, val_dataset: Optional[RatingsDataset] = None
    ) -> AbstractTrainer:
        model = NCFCombined(self.model_hyperparameters)

        model.gmf.load_state_dict(torch.load(CHECKPOINT_PATH / GMF_PRETRAINED_NAME)["model"])
        model.mlp.load_state_dict(torch.load(CHECKPOINT_PATH / MLP_PRETRAINED_NAME)["model"])

        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        evaluator = RatingEvaluator(model, BATCH_SIZE, train_dataset, val_dataset)

        return RatingTrainer(model, BATCH_SIZE, optimizer, evaluator=evaluator)


if __name__ == "__main__":
    procedure = NCFCombinedTrainingProcedure()
    procedure.start_procedure()
