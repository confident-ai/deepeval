"""Pipeline
"""
from .metrics.metric import Metric
from typing import Callable


class Pipeline:
    def __init__(self, pipeline_id: str, result_function: Callable):
        self.pipeilne_id = pipeline_id
        self.result_function = result_function
