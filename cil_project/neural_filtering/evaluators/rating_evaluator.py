import numpy as np
import torch
from cil_project.neural_filtering.evaluators.abstract_evaluator import AbstractEvaluator


class RatingEvaluator(AbstractEvaluator):
    """
    Evaluator for models that predict the ratings given (user_id, movie_id) pairs.
    """

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        torch_inputs = torch.from_numpy(inputs).to(self.device)

        output_size = inputs.shape[0]
        predictions = np.empty((output_size, 1), dtype=np.float32)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_idx = 0

            for i in range(0, output_size, self.batch_size):
                upper_bound = min(i + self.batch_size, output_size)
                y_hat = self.model(torch_inputs[i:upper_bound]).detach().cpu().numpy()
                size = y_hat.shape[0]
                predictions[start_idx : start_idx + size] = y_hat
                start_idx += size

        return predictions
