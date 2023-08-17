"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
import random
import os
import warnings
from ..constants import API_KEY_ENV, LOG_TO_SERVER_ENV
from abc import abstractmethod
from ..api import Api
from ..utils import softmax


class Metric:
    def __call__(self, *args, **kwargs):
        result = self.measure(*args, **kwargs)
        if self._is_send_okay():
            self._send_to_server(**kwargs)
        return result

    @abstractmethod
    def measure(self, a, b):
        pass

    @abstractmethod
    def is_successful(self) -> bool:
        return False

    def _is_api_key_set(self):
        result = os.getenv(API_KEY_ENV) is not None
        if result is False:
            warnings.warn(
                """API key is not set. Please set it by visiting https://app.confident-ai.com
"""
            )
        return result

    def _is_send_okay(self):
        return self._is_api_key_set() and os.getenv(LOG_TO_SERVER_ENV) == "Y"

    async def _send_to_server(self, entailment_score, input, output, **kwargs):
        client = Api(api_key=os.getenv(API_KEY_ENV))
        return client.add_test_case(
            entailment_score=entailment_score,
            output=output,
            input=input,
            metric=self.__class__.__name__,
            is_successful=self.is_successful(),
        )


class EntailmentScoreMetric(Metric):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-base"):
        # We use a smple cross encoder model
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def measure(self, a: str, b: str):
        scores = self.model.predict([(a, b)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        return softmax(scores)[0][1]
