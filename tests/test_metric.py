"""Tests for metric
"""
import os
from deepeval.metrics.metric import Metric
from deepeval.constants import API_KEY_ENV, LOG_TO_SERVER_ENV


class RandomMetric(Metric):
    def is_successful(self):
        return True

    def measure(self, a, b):
        return 0.5


def test_is_api_key_set():
    metric = RandomMetric()
    assert metric._is_api_key_set() == False


def test_is_api_key_set():
    os.environ[API_KEY_ENV] = "XYZFHEUIHFA"
    os.environ[LOG_TO_SERVER_ENV] = "ABCDEFGHI"
    metric = RandomMetric()
    assert metric._is_api_key_set() == True
