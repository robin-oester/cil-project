import pathlib

import BayesianFactorizationMachines.bayesian_factorization_machines as BFM
from dataset import BalancedKFold, RatingsDataset

if __name__ == "__main__":

    # Example usage
    dataset = RatingsDataset(file_path=pathlib.Path("./data/data_train.csv"))

    k_fold = BalancedKFold(5, True)
    iterator = k_fold.split(dataset)

    # BFM example
    # Create an instance of BayesianFactorizationMachines
    bfm = BFM.BayesianFactorizationMachine(dataset, iterator)

    # Train the model
    # bfm.train()

    # Predict a rating
    try:
        prediction = bfm.predict((1, 1))
        print(f"Predicted rating for user 1 and movie 1: {prediction}")
    except ValueError as e:
        print(e)
