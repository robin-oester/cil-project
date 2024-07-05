from cil_project.neural_filtering.evaluators import RatingEvaluator
from cil_project.neural_filtering.models import NCFBaseline
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME
from dataset import RatingsDataset

if __name__ == "__main__":
    print("Hello World!")
    # Example usage
    model = NCFBaseline.load_from_checkpoint("NCFBaseline_5_2024-06-25_10:02:46.pkl")
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

    # optional normalization
    # dataset.normalize(TargetNormalization.BY_MOVIE)

    evaluator = RatingEvaluator(model, 64, dataset, None)

    evaluator.generate_predictions(SUBMISSION_FILE_NAME)
