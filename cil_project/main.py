import pathlib

from dataset import RatingsDataset

if __name__ == "__main__":

    # Example usage
    dataset = RatingsDataset.deserialize(pathlib.Path("../data/serialized_ratings.npz"))
    print(dataset[0])
