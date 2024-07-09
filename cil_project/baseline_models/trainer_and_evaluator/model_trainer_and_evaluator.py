from cil_project.baseline_models.models import ALS, SVP, SVT, Baseline
from cil_project.dataset import BalancedKFold, RatingsDataset, SubmissionDataset
from cil_project.ensembling.utils import write_predictions_to_csv
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME

# To run the code: python3 cil_project/../model_trainer_and_evaluator.py


# np.random.seed(0)
# torch.manual_seed(0)

VERBOSE = True
MODEL_ID = 0  # 0: SVT, 1: SVP, 2: ALS


# set hyperparameters for the model here
def get_model(model_id: int) -> Baseline:
    if model_id == 0:
        return SVT(max_iter=60, eta=1.05, tau=37, verbose=VERBOSE)
    if model_id == 1:
        return SVP(max_iter=20, eta=1.3315789473684212, k=4, verbose=VERBOSE)
    if model_id == 2:
        return ALS(max_iter=20, lam=0.2, k=3, verbose=VERBOSE)
    raise ValueError("Invalid model id")


def start_kfold_training(nr_folds: int = 10) -> None:
    # prepare dataset
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    kfold = BalancedKFold(num_folds=nr_folds, shuffle=True)

    avg_rmse = 0.0  # pylint: disable=invalid-name

    # initialize model
    model = get_model(MODEL_ID)

    # KFold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1} started")
        # training split
        train_data = dataset.get_split(train_idx)
        train_matrix = train_data.get_data_matrix()
        # test split
        test_data = dataset.get_split(test_idx)
        test_matrix = test_data.get_data_matrix()
        test_matrix_mask = test_data.get_data_matrix_mask().astype(int)

        # train model
        model.train(train_matrix, test_matrix, test_matrix_mask)
        avg_rmse += model.rmse

    print(f"Average RMSE over 10 folds: {avg_rmse / 10.0}")


def generate_data_for_stacking() -> None:

    avg_rmse = 0.0  # pylint: disable=invalid-name

    for fold in range(10):
        print(f"Fold {fold+1} started")

        model = get_model(MODEL_ID)

        train_dataset = RatingsDataset.load(f"stacking_train_{fold}")
        val_dataset = RatingsDataset.load(f"stacking_val_{fold}")

        # training split
        train_matrix = train_dataset.get_data_matrix()

        # test split
        val_matrix = val_dataset.get_data_matrix()
        val_matrix_mask = val_dataset.get_data_matrix_mask().astype(int)

        # train model
        model.train(train_matrix, val_matrix, val_matrix_mask)

        #  generate predictions for validation fold
        inpts = val_dataset.get_inputs()
        preds = model.predict(inpts)
        fold_dataset = SubmissionDataset(inpts, preds)
        write_predictions_to_csv(fold_dataset, model.get_name(), fold)

        avg_rmse += model.rmse

    print(f"Average RMSE over 10 folds: {avg_rmse / 10.0}")

    # generate predictions for the test set
    model = get_model(MODEL_ID)
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    model.train(dataset.get_data_matrix())
    model.generate_predictions(SUBMISSION_FILE_NAME)


def generate_data_for_blending() -> None:

    model = get_model(MODEL_ID)

    train_dataset = RatingsDataset.load("blending_train")
    val_dataset = RatingsDataset.load("blending_val")

    # training split
    train_matrix = train_dataset.get_data_matrix()

    # test split
    val_matrix = val_dataset.get_data_matrix()
    val_matrix_mask = val_dataset.get_data_matrix_mask().astype(int)

    # train model
    model.train(train_matrix, val_matrix, val_matrix_mask)

    #  generate predictions for validation fold
    inpts = val_dataset.get_inputs()
    preds = model.predict(inpts)
    fold_dataset = SubmissionDataset(inpts, preds)
    write_predictions_to_csv(fold_dataset, model.get_name(), 0)

    # generate predictions for the test set
    model = get_model(MODEL_ID)
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)
    model.train(dataset.get_data_matrix())
    model.generate_predictions(SUBMISSION_FILE_NAME)


def main():
    start_kfold_training()
    # generate_data_for_stacking()
    # generate_data_for_blending()


if __name__ == "__main__":
    main()
