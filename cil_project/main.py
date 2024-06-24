from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from dataset import RatingsDataset

if __name__ == "__main__":

    # Example usage
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    print(dataset[0])
