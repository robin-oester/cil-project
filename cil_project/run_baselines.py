import numpy as np
from baseline_models import ALS
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME
from dataset import RatingsDataset

# from baseline_models import ALS, SVP, SVT
# from dataset.balanced_k_fold import BalancedKFold

np.random.seed(42)


def rmse(d_matrix: np.ndarray, reconstructed_matrix: np.ndarray) -> float:
    """
    Calculate the root mean squared error between the data matrix and the reconstructed matrix.
    """
    mask = d_matrix != 0
    return np.sqrt(np.mean((d_matrix[mask] - reconstructed_matrix[mask]) ** 2))


def get_matrix(data: list[tuple[tuple[int, int], float]], num_users: int = 10000, num_movies: int = 1000) -> np.ndarray:
    """
    Returns the dataset as a matrix. Each non-zero value marks an observed rating.
    """

    ratings = np.zeros((num_users, num_movies), dtype=float)

    for idx, ((user_id, movie_id), rating) in enumerate(data):
        if ratings[user_id][movie_id] > 0:
            raise ValueError(f"Duplicate rating at index '{idx}'.")
        ratings[user_id][movie_id] = rating
    return ratings


if __name__ == "__main__":
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    data_matrix = dataset.get_data_matrix()

    # --- test ALS ---
    als = ALS()
    als.train(data_matrix)
    print("prediction: ", als.predict((1, 2)))

    # examlpe of how to load model attributes
    # als = ALS.load_model_attributes("./path/to/ALS.pkl")
    # ----------------

    # --- test SVP ---
    # svp = SVP(max_iter=2)
    # svp.train(data_matrix)
    # print("prediction: ", svp.predict((1, 2)))

    # examlpe of how to load model attributes
    # svp = SVP.load_model_attributes("./path/to/SVP.pkl")
    # ----------------

    # --- test SVP ---
    # svt = SVT()
    # svt.train(data_matrix)
    # print("prediction: ", svt.predict((1, 2)))

    # examlpe of how to load model attributes
    # svt = SVT.load_model_attributes("./path/to/SVT.pkl")
    # ----------------

    # rmse_list = [0, 0, 0]
    # kfold = BalancedKFold(num_folds=10, shuffle=True)
    # svt = SVT()
    # for train_idx, test_idx in kfold.split(dataset):
    #    train_data = [dataset.data[i] for i in train_idx]
    #    test_data = [dataset.data[i] for i in test_idx]
    #    train_matrix = get_matrix(train_data)
    #    svt.train(train_matrix)
    #    reconstr = svt.reconstructed_matrix
    #    test_matrix = get_matrix(test_data)
    #    print("SVT RMSE: ", rmse(test_matrix, reconstr))
    #    rmse_list[0] += rmse(test_matrix, reconstr)
    # rmse_list[0] /= 10

    # kfold = BalancedKFold(num_folds=10, shuffle=True)
    # svp = SVP()
    # for train_idx, test_idx in kfold.split(dataset):
    #    train_data = [dataset.data[i] for i in train_idx]
    #    test_data = [dataset.data[i] for i in test_idx]
    #    train_matrix = get_matrix(train_data)
    #    svp.train(train_matrix)
    #    reconstr = svp.reconstructed_matrix
    #    test_matrix = get_matrix(test_data)
    #    print("SVP RMSE: ", rmse(test_matrix, reconstr))
    #    rmse_list[1] += rmse(test_matrix, reconstr)
    # rmse_list[1] /= 10

    # kfold = BalancedKFold(num_folds=10, shuffle=True)
    # als = ALS()
    # for train_idx, test_idx in kfold.split(dataset):
    #    train_data = [dataset.data[i] for i in train_idx]
    #    test_data = [dataset.data[i] for i in test_idx]
    #    train_matrix = get_matrix(train_data)
    #    als.train(train_matrix)
    #    reconstr = als.reconstructed_matrix
    #    test_matrix = get_matrix(test_data)
    #    print("ALS RMSE: ", rmse(test_matrix, reconstr))
    #    rmse_list[2] += rmse(test_matrix, reconstr)
    # rmse_list[2] /= 10

    # print("---------------------Balanced10fold RMSE LIST--------------------------")
    # print("RMSE for SVT: ", rmse_list[0], "RMSE for SVP: ", rmse_list[1], "RMSE for ALS: ", rmse_list[2])
    # print("-----------------------------------------------------------------------")
