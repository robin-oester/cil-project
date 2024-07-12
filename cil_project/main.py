from cil_project.bayesian_factorization_machines.training import BFMTrainingProcedure
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from dataset import RatingsDataset

if __name__ == "__main__":
    print("Hello World!")

    # Example usage
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    # test bin sizes from 2 to 1024 in powers of 2
    for i in range(1, 2):
        NUM_BINS = 10
        print(f"Training with {NUM_BINS} bins")
        trainer = BFMTrainingProcedure(
            rank=10,
            num_bins=NUM_BINS,
            iterations=500,
            kfold=10,
            dataset=dataset,
            grouped=True,
            implicit=True,
            statistical_features=False,
            ordinal_probit=True,
        )
        trainer.start_training()
    # trainer = BFMTrainingProcedure(10, 500, 1, dataset, True, True, False, True)
    # trainer.final_train()

    # trainer.model.generate_predictions(SUBMISSION_FILE_NAME)
