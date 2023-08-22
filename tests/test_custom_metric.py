"""Test for custom metrics in Python
"""

import pytest
import asyncio
from deepeval.metrics.metric import Metric


class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(self, minimum_length: int = 3):
        self.minimum_length = minimum_length

    def __call__(self, text: str):
        # sends to server
        score = self.measure(text)
        # Optional: Logs it to the server
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=text,
                success=self.success,
            )
        )
        return self.measure(text)

    def measure(self, text: str):
        score = len(text)
        self.success = score > self.minimum_length
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"


@pytest.mark.asyncio
async def test_length_metric():
    metric = LengthMetric()
    metric.measure("fehusihfe wuifheuiwh")
    assert metric.is_successful()
