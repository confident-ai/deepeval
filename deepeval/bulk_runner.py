from typing import List, Callable
from abc import abstractmethod
from tabulate import tabulate
from .metrics.metric import Metric
from .test_case import TestCase


class BulkTestRunner:
    def __init__(self, test_filename: str = None):
        self.test_filename = test_filename

    @abstractmethod
    def bulk_test_cases(self) -> List[TestCase]:
        return []

    def run(self, completion_fn: Callable):
        table = []

        headers = [
            "Test Passed",
            "Metric Name",
            "Score",
            "Output",
            "Expected output",
            "Message",
        ]
        for case in self.bulk_test_cases:
            case: TestCase
            output = completion_fn(case.query)
            for metric in case.metrics:
                score = metric(output, case.expected_output)
                is_successful = metric.is_successful()
                message = f"""{metric.__class__.__name__} was unsuccessful for 
{case.query}
which should have matched 
{case.expected_output}
"""
                table.append(
                    [
                        is_successful,
                        metric.__class__.__name__,
                        score,
                        output,
                        case.expected_output,
                        message,
                    ]
                )
        with open(self.test_filename, "w") as f:
            f.write(tabulate(table, headers=headers))
        for t in table:
            assert t[0] == True, t[-1]
        return table
