from typing import List, Union


# from deepeval.experimental import BaseEvaluationExperiment

try:
    from transformers.trainer_callback import TrainerCallback

    class DeepEvalHarnessCallback(TrainerCallback):
        """
        A [transformers.TrainerCallback] that logs various harness LLM evaluation metrics to DeepEval
        """

        def __init__(self, experiments):
            super().__init__()
            self.experiments = experiments

            raise NotImplementedError("DeepEvalHarnessCallback is WIP")

except ImportError:

    class DeepEvalHarnessCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'transformers' library is required to use the DeepEvalHarnessCallback."
            )
