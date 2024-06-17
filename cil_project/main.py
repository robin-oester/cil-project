import pathlib

from dataset import BalancedKFold, RatingsDataset
from BayesianFactorizationMachines import BFM

if __name__ == "__main__":

    # Example usage
    dataset = RatingsDataset(file_path=pathlib.Path("./data/data_train.csv"))
    print(dataset[0])

    k_fold = BalancedKFold(5, True)
    iterator = k_fold.split(dataset)

    for train_idx, test_idx in iterator:
        training_samples = len(train_idx)
        test_samples = len(test_idx)
        print(training_samples, test_samples)

    # BFM example  
    BFM.train(dataset.get_data_frame(), k_fold.split(dataset))