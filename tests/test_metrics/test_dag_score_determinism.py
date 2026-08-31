"""DAGMetric score determinism when several scoring verdicts fire.

A DAG with more than one reachable scoring verdict used to resolve to
whichever verdict wrote ``metric.score`` last, so the outcome depended on
node declaration order and on whether the graph ran in sync or async mode.
The runner now collects every fired verdict score and resolves the final
score deterministically with a conservative ``min``, so the result only
depends on the DAG as written.
"""

import asyncio

import pytest

from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

PARAMS = [SingleTurnParams.ACTUAL_OUTPUT]


class AlwaysYes(DeepEvalBaseLLM):
    """Answers True to every judgement so every scoring verdict is reached.

    ``a_generate`` yields to the event loop so async runs really interleave;
    otherwise the async parametrization silently executes serially.
    """

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        if schema.__name__ == "TaskNodeOutput":
            return schema(output=["Intro", "Body", "Conclusion"])
        return schema(verdict=True, reason="fixed")

    async def a_generate(self, prompt, schema=None, **kwargs):
        await asyncio.sleep(0)
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "always-yes"


def build_two_scoring_branches(deep_first: bool) -> DeepAcyclicGraph:
    """Two independent scoring branches, both always reached.

    ``deep`` is two judgements deep and scores 10 (1.0); ``shallow`` is one
    judgement deep and scores 5 (0.5). ``deep_first`` toggles which one is
    declared first so tests can prove the result does not depend on it.
    """
    extract = TaskNode(
        instructions="Extract the headings",
        output_label="Headings",
        evaluation_params=PARAMS,
    )
    deep = BinaryJudgementNode(criteria="All three headings present?")
    inner = BinaryJudgementNode(criteria="Headings in the right order?")
    shallow = BinaryJudgementNode(criteria="Summary under 100 words?")

    if deep_first:
        extract.add_node(deep)
        extract.add_node(shallow)
    else:
        extract.add_node(shallow)
        extract.add_node(deep)

    deep.add_verdict(True, then=inner)
    deep.add_verdict(False, score=0)
    inner.add_verdict(True, score=10)
    inner.add_verdict(False, score=0)
    shallow.add_verdict(True, score=5)
    shallow.add_verdict(False, score=0)
    return DeepAcyclicGraph(root_nodes=[extract])


def build_single_scoring_branch() -> DeepAcyclicGraph:
    """A DAG with exactly one scoring verdict, for the no-regression case."""
    extract = TaskNode(
        instructions="Extract the headings",
        output_label="Headings",
        evaluation_params=PARAMS,
    )
    shallow = BinaryJudgementNode(criteria="Summary under 100 words?")
    extract.add_node(shallow)
    shallow.add_verdict(True, score=10)
    shallow.add_verdict(False, score=0)
    return DeepAcyclicGraph(root_nodes=[extract])


def measure(
    dag: DeepAcyclicGraph, async_mode: bool, include_reason: bool = False
) -> DAGMetric:
    metric = DAGMetric(
        name="Score Determinism",
        dag=dag,
        model=AlwaysYes(model="always-yes"),
        include_reason=include_reason,
        async_mode=async_mode,
    )
    metric.measure(
        LLMTestCase(
            input="Summarize.",
            actual_output="Intro\nBody\nConclusion",
        ),
        _show_indicator=False,
    )
    return metric


class TestMultiScoringVerdicts:
    """Two reachable scoring verdicts resolve deterministically."""

    def test_sync_and_async_agree(self):
        scores = [
            measure(build_two_scoring_branches(deep_first), async_mode).score
            for deep_first in (True, False)
            for async_mode in (False, True)
        ]
        assert len(set(scores)) == 1

    def test_score_is_conservative_min_of_fired_verdicts(self):
        # deep scores 1.0, shallow scores 0.5; min wins
        metric = measure(
            build_two_scoring_branches(deep_first=True), async_mode=False
        )
        assert metric.score == 0.5

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_declaration_order_irrelevant(self, async_mode):
        assert (
            measure(
                build_two_scoring_branches(deep_first=True), async_mode
            ).score
            == measure(
                build_two_scoring_branches(deep_first=False), async_mode
            ).score
        )

    def test_reason_tracks_selected_score(self):
        metric = measure(
            build_two_scoring_branches(deep_first=True),
            async_mode=False,
            include_reason=True,
        )
        assert metric.score == 0.5
        assert metric.reason == "fixed"


class TestSingleScoringVerdict:
    """A DAG with one scoring verdict is unaffected (no regression)."""

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_score_unchanged(self, async_mode):
        assert measure(build_single_scoring_branch(), async_mode).score == 1.0
