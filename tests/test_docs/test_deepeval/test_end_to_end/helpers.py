from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, List, Tuple

from deepeval.dataset import Golden, EvaluationDataset
from deepeval.dataset.golden import ConversationalGolden
from deepeval.metrics import BaseMetric, BaseConversationalMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    Turn,
)


def deterministic_llm_app(user_input: str) -> Tuple[str, List[str]]:
    """
    Deterministic stand-in for "your_llm_app" from the docs.

    The docs show:
        res, text_chunks = your_llm_app(golden.input)

    We return:
      - res: deterministic output based solely on input
      - text_chunks: deterministic retrieval_context
    """
    normalized = user_input.strip().lower()
    if "name" in normalized:
        return "My name is DeepEval.", ["ctx: identity", "ctx: greeting"]
    if "number" in normalized:
        return "42", ["ctx: numbers", "ctx: preferences"]
    return "OK", ["ctx: default"]


def build_single_turn_dataset() -> EvaluationDataset:
    # Mirrors the docs pattern (goldens list -> EvaluationDataset(goldens))
    goldens = [
        Golden(
            input="What is your name?", expected_output="My name is DeepEval."
        ),
        Golden(input="Choose a number between 1 to 100", expected_output="42"),
    ]
    return EvaluationDataset(goldens)


def build_llm_test_cases_from_goldens(
    dataset: EvaluationDataset,
    llm_app: Callable[[str], Tuple[str, List[str]]] = deterministic_llm_app,
) -> List[LLMTestCase]:
    test_cases: List[LLMTestCase] = []
    for golden in dataset.goldens:
        res, text_chunks = llm_app(golden.input)
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=res,
                expected_output=golden.expected_output,
                retrieval_context=text_chunks,
            )
        )
    return test_cases


class DeterministicContainsExpectedOutputMetric(BaseMetric):
    """
    Tiny deterministic metric for offline CI.
    Avoid asserting exact metric scores in tests; we only need stable behavior.
    """

    _required_params = [
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "DeterministicContainsExpectedOutputMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = (test_case.expected_output or "").strip()
        actual = (test_case.actual_output or "").strip()
        passed = (expected != "") and (expected in actual)
        self.score = 1.0 if passed else 0.0
        self.reason = (
            "expected_output is contained in actual_output"
            if passed
            else "expected_output not found in actual_output"
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.score is not None and self.score >= self.threshold)


class DeterministicFailingMetric(BaseMetric):
    """
    Deterministic metric that always fails.
    Used to verify that evaluate() correctly propagates failures.
    """

    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "DeterministicFailingMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = 0.0
        self.reason = "This metric always fails for testing purposes"
        self.success = False
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return False


class DeterministicPassingMetric(BaseMetric):
    """
    Deterministic metric that always passes.
    Used to verify that evaluate() correctly propagates success.
    """

    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "DeterministicPassingMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = 1.0
        self.reason = "This metric always passes for testing purposes"
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return True


def save_dataset_as_json_and_load(
    dataset: EvaluationDataset, directory: Path, file_name: str
) -> list:
    """
    Option A artifact: dataset.save_as(file_type="json", directory=..., file_name=...)
    Returns parsed JSON content (a list of records).
    """
    full_path = dataset.save_as(
        file_type="json",
        directory=str(directory),
        file_name=file_name,
        include_test_cases=False,
    )
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_as_csv_and_load(
    dataset: EvaluationDataset, directory: Path, file_name: str
) -> List[dict]:
    """
    Option A artifact: dataset.save_as(file_type="csv", directory=..., file_name=...)
    Returns parsed CSV content as a list of dicts.
    """
    full_path = dataset.save_as(
        file_type="csv",
        directory=str(directory),
        file_name=file_name,
        include_test_cases=False,
    )
    with open(full_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ===========================================================================
# Multi-turn / Conversational helpers
# ===========================================================================


def deterministic_chatbot_callback(
    input: str,
    turns: List[Turn] = None,
    thread_id: str = None,
) -> Turn:
    """
    Deterministic chatbot callback for offline testing.

    Mirrors the doc pattern:
        async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
            response = await your_chatbot(input, turns, thread_id)
            return Turn(role="assistant", content=response)

    This sync version returns deterministic responses based on input.
    """
    normalized = input.strip().lower()
    if (
        "ticket" in normalized
        or "buy" in normalized
        or "purchase" in normalized
    ):
        return Turn(
            role="assistant",
            content="I can help you purchase a ticket. What event are you interested in?",
        )
    if "coldplay" in normalized or "concert" in normalized:
        return Turn(
            role="assistant",
            content="Great choice! We have VIP and standard tickets available for Coldplay.",
        )
    if "vip" in normalized:
        return Turn(
            role="assistant",
            content="VIP ticket selected. That will be $250. Shall I proceed with the purchase?",
        )
    if (
        "yes" in normalized
        or "proceed" in normalized
        or "confirm" in normalized
    ):
        return Turn(
            role="assistant",
            content="Purchase confirmed! Your VIP ticket has been booked successfully.",
        )
    return Turn(role="assistant", content="How can I assist you today?")


def build_multi_turn_dataset() -> EvaluationDataset:
    """
    Build a multi-turn dataset using ConversationalGolden.

    Mirrors the docs pattern:
        goldens = [
            ConversationalGolden(
                scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
                expected_outcome="Successful purchase of a ticket.",
                user_description="Andy Byron is the CEO of Astronomer.",
            ),
            ...
        ]
        dataset = EvaluationDataset(goldens)
    """
    goldens = [
        ConversationalGolden(
            scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
            expected_outcome="Successful purchase of a ticket.",
            user_description="Andy Byron is the CEO of Astronomer.",
        ),
        ConversationalGolden(
            scenario="A customer wants to ask about concert dates.",
            expected_outcome="Customer receives concert date information.",
            user_description="A general user looking for event info.",
        ),
    ]
    return EvaluationDataset(goldens)


def build_conversational_test_cases_manually(
    dataset: EvaluationDataset,
    chatbot_callback: Callable = deterministic_chatbot_callback,
    max_turns: int = 4,
) -> List[ConversationalTestCase]:
    """
    Manually build ConversationalTestCase objects without using ConversationSimulator.

    ConversationSimulator requires a simulator_model which needs network access.
    This helper creates deterministic test cases for offline testing.
    """
    test_cases = []
    for golden in dataset.goldens:
        # Simulate a basic conversation flow
        turns = []
        user_inputs = [
            "Hello, I want to buy a ticket",
            "I'm interested in Coldplay",
            "I'd like a VIP ticket please",
            "Yes, please proceed",
        ]

        for i, user_input in enumerate(user_inputs[:max_turns]):
            # User turn
            turns.append(Turn(role="user", content=user_input))
            # Assistant response via callback
            assistant_turn = chatbot_callback(
                user_input, turns, f"thread-{id(golden)}"
            )
            turns.append(assistant_turn)

        test_case = ConversationalTestCase(
            turns=turns,
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
        )
        test_cases.append(test_case)

    return test_cases


class DeterministicConversationalMetric(BaseConversationalMetric):
    """
    Deterministic conversational metric for offline testing.
    Evaluates whether the conversation reached an expected outcome.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "DeterministicConversationalMetric"

    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        # Check if any assistant turn contains expected outcome keywords
        outcome_keywords = ["confirmed", "booked", "success", "complete"]
        has_positive_outcome = any(
            any(kw in turn.content.lower() for kw in outcome_keywords)
            for turn in test_case.turns
            if turn.role == "assistant"
        )
        self.score = 1.0 if has_positive_outcome else 0.0
        self.reason = (
            "Conversation reached a positive outcome"
            if has_positive_outcome
            else "Conversation did not reach expected outcome"
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.score is not None and self.score >= self.threshold)
