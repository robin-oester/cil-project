# import csv
import pathlib

from als import ALS
from dataset.ratings_dataset import RatingsDataset

if __name__ == "__main__":
    dataset = RatingsDataset(pathlib.Path("./data/data_train.csv"))
    data_matrix = dataset.get_data_matrix()
    als = ALS()
    als.train(data_matrix)
    # test prediction
    print("prediction: ", als.predict((1, 2)))
