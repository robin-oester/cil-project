from cil_project.bayesian_factorization_machines.training import BFMTrainingProcedure
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from dataset import RatingsDataset

if __name__ == "__main__":
    print("Hello World!")

    # Example usage
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    # test bin sizes from 2 to 1024 in powers of 2
    for i in range(1, 9):
        NUM_BINS = 2**i
        NUM_CLUSTERS = 2**i
        print(f"Training with {NUM_BINS} bins")
        # print(f"Training with {NUM_CLUSTERS} clusters")
        trainer = BFMTrainingProcedure(
            rank=15,
            num_bins=NUM_BINS,
            num_clusters=NUM_CLUSTERS,
            iterations=1000,
            kfold=10,
            dataset=dataset,
            grouped=False,
            implicit=False,
            statistical_features=True,
            ordinal_probit=False,
            kmeans=False,
        )
        trainer.start_training()

    # for final train
    # trainer = BFMTrainingProcedure(
    #     rank=2,
    #     num_bins=10,
    #     iterations=100,
    #     kfold=10,
    #     dataset=dataset,
    #     grouped=True,
    #     implicit=True,
    #     statistical_features=False,
    #     ordinal_probit=False,
    #     kmeans=False,
    # )
    # trainer.final_train()

    # trainer.model.generate_predictions(SUBMISSION_FILE_NAME)
