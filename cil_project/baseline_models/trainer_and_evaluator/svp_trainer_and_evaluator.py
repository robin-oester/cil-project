from cil_project.baseline_models.models.svp import SVP
from cil_project.dataset import BalancedKFold, RatingsDataset
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME, masked_rmse

ONLY_GENERATE_SUBMISSION = False
VERBOSE = True

# hyperparameters
MAX_ITER = 60
ETA = 0.8
K = 5

# To run the code: python3 cil_project/../svp_trainer_and_evaluator.py
if __name__ == "__main__":
    # prepare dataset
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    kfold = BalancedKFold(num_folds=10, shuffle=True)

    # initialize SVP model
    svp = SVP(max_iter=MAX_ITER, eta=ETA, k=K, verbose=VERBOSE)

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
        svp.train(train_matrix)
        reconstr = svp.reconstructed_matrix

        # validate model
        rmse = masked_rmse(test_matrix, reconstr, test_matrix_mask)
        print(f"Fold {fold+1} validation RMSE: {rmse}")

    # train on the whole dataset and generate submission file
    print("Training on the whole dataset")
    svp.train(dataset.get_data_matrix())
    print("Generating submission file")
    svp.generate_predictions(SUBMISSION_FILE_NAME)