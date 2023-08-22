"""Available metrics. The best metric that
you want is Cohere's reranker metric.
"""
import asyncio
import os
import warnings
from typing import Optional
from ..constants import API_KEY_ENV, IMPLEMENTATION_ID_ENV, LOG_TO_SERVER_ENV
from abc import abstractmethod
from ..api import Api
from ..utils import softmax


class Metric:
    @abstractmethod
    def measure(self, output, expected_output, query: Optional[str] = None):
        pass

    def _get_init_values(self):
        # We use this method for sending useful metadata
        init_values = {
            param: getattr(self, param)
            for param in vars(self)
            if isinstance(getattr(self, param), (str, int, float))
        }
        return init_values

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
        # DOing this until the API endpoint is fixed
        return self._is_api_key_set() and os.getenv(LOG_TO_SERVER_ENV) != "Y"

    def __call__(self, output, expected_output, query: Optional[str] = "-"):
        score = self.measure(output, expected_output)
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=query,
                output=output,
            )
        )
        return score

    async def _send_to_server(
        self,
        metric_score: float,
        metric_name: str,
        query: str = "-",
        output: str = "-",
        expected_output: str = "-",
        implementation_id: str = None,
        **kwargs
    ):
        if self._is_send_okay():
            client = Api(api_key=os.getenv(API_KEY_ENV))
            if implementation_id is None:
                implementation_id = os.getenv(IMPLEMENTATION_ID_ENV)
            datapoint_id = client.add_golden(
                query=query,
                expected_output=expected_output,
            )
            return client.add_test_case(
                metric_score=float(metric_score),
                metric_name=metric_name,
                actual_output=output,
                query=query,
                implementation_id=implementation_id,
                metrics_metadata=self._get_init_values(),
                success=bool(self.is_successful()),
                datapoint_id=datapoint_id["id"],
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
