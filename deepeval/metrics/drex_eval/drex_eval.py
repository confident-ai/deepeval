"""DrexEval: ``JevEval`` judged by Nace's Drex instead of TypeSafe's Jev.

The metric is the same: the user defines the decision points as ``Noul`` /
``Score`` / ``Choice`` questions, Drex answers them in one ``decide()`` call
with calibrated probabilities, each answer maps onto ``[0, 1]`` and the score
is their weighted mean. No LLM is involved; the reason is deterministic text
built from Drex's answers.

What Drex adds is images: on a multimodal ``LLMTestCase`` the images go to
Drex Max alongside the state, which refers to them as ``[Image 1]``,
``[Image 2]``, ... (the default ``drex-max-v1.1`` reads them; a text-only
model refuses the test case rather than dropping the images).
"""

from typing import Optional, Union

from deepeval.metrics.jev_eval.jev_eval import JevEval
from deepeval.metrics.jev_eval.utils import initialize_drex_model
from deepeval.models import DeepEvalBaseSystemOneModel


class DrexEval(JevEval):
    _suffix = "DrexEval"

    def _initialize_system_one_model(
        self, model: Optional[Union[str, DeepEvalBaseSystemOneModel]]
    ) -> DeepEvalBaseSystemOneModel:
        return initialize_drex_model(model)
