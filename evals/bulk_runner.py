from typing import List, Callable
from abc import abstractmethod
from .metrics.metric import Metric
from .test_case import TestCase


class BulkTestRunner:
    def __init__(self):
        pass

    @abstractmethod
    def bulk_test_cases(self) -> List[TestCase]:
        return []

    def run(self, callable_fn: Callable):
        for case in self.bulk_test_cases:
            case: TestCase
            output = callable_fn(case.input)
            for metric in case.metrics:
                output = metric(case.input, case.expected_output)
        assert output
