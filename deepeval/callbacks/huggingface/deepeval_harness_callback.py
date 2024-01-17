from typing import List, Union

from transformers.trainer_callback import TrainerCallback

# from deepeval.experimental import BaseEvaluationExperiment


class DeepEvalHarnessCallback(TrainerCallback):
    """
    A [transformers.TrainerCallback] that logs various harness LLM evaluation metrics to DeepEval
    """

    def __init__(self, experiments):
        super().__init__()
        self.experiments = experiments

        raise NotImplementedError("DeepEvalHarnessCallback is WIP")
