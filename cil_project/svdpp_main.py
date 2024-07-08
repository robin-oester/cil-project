from cil_project.svd_plusplus.evaluators import SVDPPEvaluator
from cil_project.svd_plusplus.model import SVDPP
from cil_project.utils import FULL_SERIALIZED_DATASET_NAME, SUBMISSION_FILE_NAME
from dataset import RatingsDataset

if __name__ == "__main__":
    print("Hello World!")

    # Example usage
    model = SVDPP.load_from_checkpoint(
        "/Users/pieroneri/Desktop/cil-project/cil_project/svd_plusplus/"
        "trainer/checkpoints/SVDPP_5_2024-07-08_21:39:06.pkl"
    )
    dataset = RatingsDataset.load(FULL_SERIALIZED_DATASET_NAME)

    evaluator = SVDPPEvaluator(model, 64, dataset, None)

    evaluator.generate_predictions(SUBMISSION_FILE_NAME)
