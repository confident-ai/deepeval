from dataclasses import dataclass
from typing import List, Dict
from deepeval.test_case import (
    LLMTestCase,
)


@dataclass
class ArenaTestCase:
    contestants: Dict[str, LLMTestCase]

    def __post_init__(self):
        contestant_names = list(self.contestants.keys())
        if len(contestant_names) != len(set(contestant_names)):
            raise ValueError("All contestant names must be unique.")

        cases = list(self.contestants.values())
        ref_input = cases[0].input
        for case in cases[1:]:
            if case.input != ref_input:
                raise ValueError("All contestants must have the same 'input'.")

        ref_expected = cases[0].expected_output
        for case in cases[1:]:
            if case.expected_output != ref_expected:
                raise ValueError(
                    "All contestants must have the same 'expected_output'."
                )


class Arena:
    test_cases: List[ArenaTestCase]
