import pathlib

from baseline_models import ALS, SVP
from dataset.ratings_dataset import RatingsDataset

if __name__ == "__main__":
    dataset = RatingsDataset(pathlib.Path("./data/data_train.csv"))
    data_matrix = dataset.get_data_matrix()

    # --- test ALS ---
    als = ALS()
    als.train(data_matrix)
    print("prediction: ", als.predict((1, 2)))

    # examlpe of how to load model attributes
    # als = ALS.load_model_attributes("./path/to/ALS.pkl")
    # ----------------

    # --- test SVP ---
    svp = SVP()
    svp.train(data_matrix)
    print("prediction: ", svp.predict((1, 2)))

    # examlpe of how to load model attributes
    # svp = SVP.load_model_attributes("./path/to/SVP.pkl")
    # ----------------
