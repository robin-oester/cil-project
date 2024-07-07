from cil_project.bayesian_factorization_machines.training import BFMTrainingProcedure
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME
from dataset import RatingsDataset

if __name__ == "__main__":
    print("Hello World!")

    # Example usage
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    trainer = BFMTrainingProcedure(10, 500, 1, dataset, True, True, False, True)
    trainer.final_train()

    trainer.model.generate_predictions(SUBMISSION_FILE_NAME)
