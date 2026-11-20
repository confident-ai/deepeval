import pytest

from deepeval.dataset import (
    EvaluationDataset,
    Golden,
    ConversationalGolden,
)
from deepeval.test_case import Turn


class TestDatasetRank:
    def test_constructor_assigns_sequential_golden_ranks(self):
        """Goldens built through the constructor get 0-based ranks."""
        goldens = [
            Golden(
                input=f"input {i}",
                actual_output="actual",
                expected_output="expected",
            )
            for i in range(3)
        ]

        dataset = EvaluationDataset(goldens=goldens)

        assert [g._dataset_rank for g in dataset.goldens] == [0, 1, 2]

    def test_goldens_setter_assigns_sequential_golden_ranks(self):
        """Goldens assigned through the `goldens` setter get 0-based ranks."""
        goldens = [
            Golden(
                input=f"input {i}",
                actual_output="actual",
                expected_output="expected",
            )
            for i in range(3)
        ]

        dataset = EvaluationDataset()
        dataset.goldens = goldens

        assert [g._dataset_rank for g in dataset.goldens] == [0, 1, 2]

    def test_constructor_assigns_sequential_conversational_golden_ranks(
        self,
    ):
        """ConversationalGoldens built through the constructor get 0-based ranks."""
        goldens = [
            ConversationalGolden(
                scenario=f"scenario {i}",
                turns=[Turn(role="user", content="hi")],
            )
            for i in range(3)
        ]

        dataset = EvaluationDataset(goldens=goldens)

        assert [g._dataset_rank for g in dataset.goldens] == [0, 1, 2]

    def test_goldens_setter_assigns_sequential_conversational_golden_ranks(
        self,
    ):
        """ConversationalGoldens assigned via the setter get 0-based ranks."""
        goldens = [
            ConversationalGolden(
                scenario=f"scenario {i}",
                turns=[Turn(role="user", content="hi")],
            )
            for i in range(3)
        ]

        dataset = EvaluationDataset()
        dataset.goldens = goldens

        assert [g._dataset_rank for g in dataset.goldens] == [0, 1, 2]

    def test_goldens_setter_ranks_match_list_positions_after_invalid_entry(
        self,
    ):
        """Ranks stay 0-based even when the setter's rollback path is unused.

        A second assignment must re-rank from 0, not accumulate stale counts.
        """
        first = Golden(input="first", actual_output="a", expected_output="e")
        dataset = EvaluationDataset()
        dataset.goldens = [first]

        second = [
            Golden(
                input=f"input {i}",
                actual_output="actual",
                expected_output="expected",
            )
            for i in range(2)
        ]
        dataset.goldens = second

        assert [g._dataset_rank for g in dataset.goldens] == [0, 1]

    def test_test_cases_ranks_unchanged(self):
        """Guard: the already-correct test case ranking keeps its behavior."""
        from deepeval.test_case import LLMTestCase

        test_cases = [
            LLMTestCase(input=f"input {i}", actual_output="actual")
            for i in range(3)
        ]

        dataset = EvaluationDataset()
        dataset.test_cases = test_cases

        assert [t._dataset_rank for t in dataset.test_cases] == [0, 1, 2]
