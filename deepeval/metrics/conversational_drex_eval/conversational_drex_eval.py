"""Multi-turn counterpart of ``DrexEval``: ``ConversationalJevEval`` judged by
Nace's Drex. Turns that carry images are judged by Drex Max."""

from typing import Optional, Union

from deepeval.metrics.conversational_jev_eval.conversational_jev_eval import (
    ConversationalJevEval,
)
from deepeval.metrics.jev_eval.utils import initialize_drex_model
from deepeval.models import DeepEvalBaseSystemOneModel


class ConversationalDrexEval(ConversationalJevEval):
    _suffix = "Conversational DrexEval"

    def _initialize_system_one_model(
        self, model: Optional[Union[str, DeepEvalBaseSystemOneModel]]
    ) -> DeepEvalBaseSystemOneModel:
        return initialize_drex_model(model)
