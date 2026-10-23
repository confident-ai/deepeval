"""What the confident-trace tests share beyond fixtures."""

from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

MEASURED: List[LLMTestCase] = []


class RecordingMetric(BaseMetric):
    threshold = 0.5

    def measure(self, test_case, *args, **kwargs):
        MEASURED.append(test_case)
        self.score, self.success = 1.0, True
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return True

    @property
    def __name__(self):
        return "Recording"


def exported(exporter):
    import confident_trace

    confident_trace.flush()
    return [span.name for span in exporter.get_finished_spans()]


def names(spans):
    return [span.name for span in spans]


def names_of(api_spans):
    return [api_span["name"] for api_span in api_spans]
