from typing import Optional, Union

from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import BaseConversationalMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type



class KnowledgeRetentionMetric(BaseConversationalMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason

    def measure(self, test_case: ConversationalTestCase):
        if len(test_case.messages) == 0:
            raise ValueError("Messages cannot be empty")
        with metrics_progress_context(self.__name__, self.evaluation_model):

            capture_metric_type(self.__name__)
            return self.score


    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Knowledge Retention"
