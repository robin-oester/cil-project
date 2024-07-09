import os

import numpy as np
import torch
from cil_project.baseline_models.models.svt import SVT
from cil_project.dataset import BalancedKFold, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME

STORE_KFOLD = False
if STORE_KFOLD:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)


ONLY_GENERATE_SUBMISSION = False
VERBOSE = True

# hyperparameters
MAX_ITER = 60
ETA = 1.05
TAU = 37


# To run the code: python3 cil_project/../svt_trainer_and_evaluator.py
if __name__ == "__main__":
    # prepare dataset
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    kfold = BalancedKFold(num_folds=10, shuffle=True)

    avg_rmse = 0.0  # pylint: disable=invalid-name

    # initialize SVT model
    svt = SVT(max_iter=MAX_ITER, eta=ETA, tau=TAU, verbose=VERBOSE)

    # KFold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        if ONLY_GENERATE_SUBMISSION:
            break
        print(f"Fold {fold+1} started")
        # training split
        train_data = dataset.get_split(train_idx)
        train_matrix = train_data.get_data_matrix()
        # test split
        test_data = dataset.get_split(test_idx)
        test_matrix = test_data.get_data_matrix()
        test_matrix_mask = test_data.get_data_matrix_mask().astype(int)

        # train model
        svt.train(train_matrix, test_matrix, test_matrix_mask)
        avg_rmse += svt.rmse

        if STORE_KFOLD:
            print("length of test dataset: ", len(test_data))
            preds = svt.predict(test_data.get_inputs())
            file_path = os.path.join(os.path.dirname(__file__), "svt_kfold_results.txt")
            with open(file_path, "ab") as f:
                np.savetxt(f, preds, fmt="%f")

    print(f"Average RMSE over 10 folds: {avg_rmse / 10.0}")

    # train on the whole dataset and generate submission file
    print("Training on the whole dataset")
    svt.train(dataset.get_data_matrix())
    print("Generating submission file")
    svt.generate_predictions(SUBMISSION_FILE_NAME)