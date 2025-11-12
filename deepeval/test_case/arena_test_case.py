from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel

from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.prompt import Prompt


class Contestant(BaseModel):
    name: str
    test_case: LLMTestCase
    hyperparameters: Optional[Dict[str, Union[str, int, float, Prompt]]] = None

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class ArenaTestCase:
    contestants: List[Contestant]

    def __post_init__(self):
        contestant_names = [contestant.name for contestant in self.contestants]
        if len(contestant_names) != len(set(contestant_names)):
            raise ValueError("All contestant names must be unique.")

        cases = [contestant.test_case for contestant in self.contestants]
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
