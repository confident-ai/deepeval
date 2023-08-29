import os
import warnings
from typing import Optional
from ..get_api_key import _get_api_key
from ..constants import (
    API_KEY_ENV,
    IMPLEMENTATION_ID_ENV,
    LOG_TO_SERVER_ENV,
    IMPLEMENTATION_ID_NAME,
)
from abc import abstractmethod
from ..client import Client
from ..utils import softmax
from ..singleton import Singleton


class Metric(metaclass=Singleton):
    # set an arbitrary minimum score that will get over-ridden later
    minimum_score: float = 0

    # Measure function signature is subject to be different - not sure
    # how applicable this is - might need a better abstraction
    @abstractmethod
    def measure(self, output, expected_output, query: Optional[str] = None):
        raise NotImplementedError

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
        raise NotImplementedError

    def _is_api_key_set(self):
        result = _get_api_key()
        # if result == "" or result is None:
        #     warnings.warn(
        #         """API key is not set - you won't be able to log to the DeepEval dashboard. Please set it by running `deepeval login`"""
        #     )
        if result == "" or result is None:
            return False
        return True

    def _is_send_okay(self):
        # DOing this until the API endpoint is fixed
        return self._is_api_key_set() and os.getenv(LOG_TO_SERVER_ENV) != "Y"

    def __call__(self, *args, **kwargs):
        score = self.measure(*args, **kwargs)
        success = score >= self.minimum_score
        self._send_to_server(
            metric_score=score,
            metric_name=self.__name__,
            query=kwargs.get("query", "-"),
            output=kwargs.get("output", "-"),
            expected_output=kwargs.get("expected_output", "-"),
            success=success,
        )
        return score

    def log(
        self,
        success: bool = True,
        score: float = 1e-10,
        metric_name: str = "-",
        query: str = "-",
        output: str = "-",
        expected_output: str = "-",
        metadata: Optional[dict] = None,
    ):
        """Log to the server.

        Parameters
        - query: What was asked to the model. This can also be context.
        - output: The LLM output.
        - expected_output: The output that's expected.
        """
        if self._is_send_okay():
            self._send_to_server(
                metric_score=score,
                metric_name=metric_name,
                query=query,
                output=output,
                expected_output=expected_output,
                success=success,
                metadata=metadata,
            )

    def _send_to_server(
        self,
        metric_score: float,
        metric_name: str,
        query: str = "-",
        output: str = "-",
        expected_output: str = "-",
        success: Optional[bool] = None,
        metadata: Optional[dict] = None,
        **kwargs
    ):
        if self._is_send_okay():
            client = Client(api_key=os.getenv(API_KEY_ENV))
            implementation_name = os.getenv(IMPLEMENTATION_ID_NAME)
            # implementation_id = os.getenv(IMPLEMENTATION_ID_ENV, "")
            # if implementation_id != "":
            implementation_id = client.get_implementation_id_by_name(
                implementation_name
            )
            os.environ[IMPLEMENTATION_ID_ENV] = implementation_id
            datapoint_id = client.add_golden(
                query=query,
                expected_output=expected_output,
            )
            if success is None:
                success = bool(self.is_successful())
                print({"success": success, "og": self.is_successful()})

            metric_metadata: dict = self._get_init_values()
            if metadata:
                metric_metadata.update(metadata)

            return client.add_test_case(
                metric_score=float(metric_score),
                metric_name=metric_name,
                actual_output=output,
                query=query,
                implementation_id=implementation_id,
                metrics_metadata=metric_metadata,
                success=bool(success),
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
