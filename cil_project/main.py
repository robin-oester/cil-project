import pathlib

from dataset.ratings_dataset import RatingsDataset

if __name__ == "__main__":

    # Example usage
    dataset = RatingsDataset(file_path=pathlib.Path("../data/data_train.csv"))
    print(dataset[0])
