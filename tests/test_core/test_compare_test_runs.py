import json
from pathlib import Path
from typing import Any

import pytest

from deepeval import compare_test_runs, judge
from deepeval.errors import IncompatibleTestRunsError, JudgeEvaluationError
from deepeval.evaluate.judge.compare_runs import a_compare_test_runs
from deepeval.evaluate.judge.schema import (
    JudgeVerdict,
    TestRunComparisonResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_run.api import LLMApiTestCase
from deepeval.test_run.test_run import TestRun

TestRun.__test__ = False


class MockJudge(DeepEvalBaseLLM):
    """Deterministic mock judge model for offline unit testing."""

    def __init__(
        self,
        verdicts: list[JudgeVerdict | str | dict] | None = None,
        should_fail: bool = False,
    ):
        self.verdicts = verdicts or []
        self.call_count = 0
        self.prompts_received = []
        self.should_fail = should_fail
        super().__init__(model="mock-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, *args, schema=None, **kwargs) -> Any:
        self.prompts_received.append(prompt)
        if self.should_fail:
            raise RuntimeError("Underlying LLM service unavailable")

        if self.call_count < len(self.verdicts):
            verdict = self.verdicts[self.call_count]
            self.call_count += 1
            return verdict

        return JudgeVerdict(winner="tie", reason="Default tie verdict")

    async def a_generate(
        self, prompt: str, *args, schema=None, **kwargs
    ) -> Any:
        return self.generate(prompt, *args, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "mock-judge"


def _create_sample_run(
    identifier: str,
    outputs: list[str],
    inputs: list[str] | None = None,
) -> TestRun:
    test_inputs = inputs or [f"Question {i+1}" for i in range(len(outputs))]
    cases = [
        LLMApiTestCase(
            name=f"case_{i+1}",
            input=test_inputs[i],
            actual_output=outputs[i],
            expected_output=f"Expected answer {i+1}",
        )
        for i in range(len(outputs))
    ]
    return TestRun(
        identifier=identifier,
        testCases=cases,
    )


def test_candidate_run_wins():
    baseline_run = _create_sample_run(
        identifier="v1.0",
        outputs=["Short reply 1", "Vague answer 2"],
    )
    candidate_run = _create_sample_run(
        identifier="v2.0",
        outputs=["Detailed accurate reply 1", "Precise factual answer 2"],
    )

    judge_model = MockJudge(
        verdicts=[
            JudgeVerdict(
                winner="candidate", reason="Candidate is more detailed."
            ),
            JudgeVerdict(
                winner="candidate", reason="Candidate is more accurate."
            ),
        ]
    )

    result: TestRunComparisonResult = compare_test_runs(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "candidate"
    assert result.candidate_wins == 2
    assert result.baseline_wins == 0
    assert result.ties == 0
    assert len(result.comparisons) == 2
    assert result.comparisons[0].winner == "candidate"
    assert "Candidate won 2 out of 2" in result.reason
    assert result.baseline_run.identifier == "v1.0"
    assert result.candidate_run.identifier == "v2.0"


def test_baseline_run_wins():
    baseline_run = _create_sample_run(
        identifier="v1.0",
        outputs=["High quality output 1", "High quality output 2"],
    )
    candidate_run = _create_sample_run(
        identifier="v2.0",
        outputs=["Hallucinated output 1", "Incomplete output 2"],
    )

    judge_model = MockJudge(
        verdicts=[
            JudgeVerdict(winner="baseline", reason="Baseline is faithful."),
            JudgeVerdict(winner="baseline", reason="Baseline is complete."),
        ]
    )

    result = compare_test_runs(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "baseline"
    assert result.baseline_wins == 2
    assert result.candidate_wins == 0
    assert result.ties == 0
    assert "Baseline won 2 out of 2" in result.reason


def test_tie_or_indeterminate_result():
    baseline_run = _create_sample_run(
        identifier="v1.0",
        outputs=["Output A", "Output B"],
    )
    candidate_run = _create_sample_run(
        identifier="v2.0",
        outputs=["Output A", "Output B"],
    )

    judge_model = MockJudge(
        verdicts=[
            JudgeVerdict(winner="tie", reason="Responses are identical."),
            JudgeVerdict(winner="tie", reason="Equal performance."),
        ]
    )

    result = compare_test_runs(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "tie"
    assert result.baseline_wins == 0
    assert result.candidate_wins == 0
    assert result.ties == 2
    assert "tied" in result.reason.lower()


def test_incompatible_different_case_counts():
    baseline_run = _create_sample_run(
        identifier="v1.0",
        outputs=["Out 1", "Out 2"],
    )
    candidate_run = _create_sample_run(
        identifier="v2.0",
        outputs=["Out 1"],
    )

    with pytest.raises(IncompatibleTestRunsError) as exc_info:
        compare_test_runs(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            judge_model=MockJudge(),
        )

    assert "Test case count mismatch" in str(exc_info.value)


def test_incompatible_mismatched_test_cases():
    baseline_run = _create_sample_run(
        identifier="v1.0",
        inputs=["What is AI?"],
        outputs=["Artificial Intelligence"],
    )
    candidate_run = _create_sample_run(
        identifier="v2.0",
        inputs=["What is Quantum Computing?"],
        outputs=["Quantum bits"],
    )

    with pytest.raises(IncompatibleTestRunsError) as exc_info:
        compare_test_runs(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            judge_model=MockJudge(),
        )

    assert "Test runs evaluate different test cases" in str(exc_info.value)


def test_empty_test_runs():
    empty_run_a = TestRun(identifier="empty_a", test_cases=[])
    empty_run_b = TestRun(identifier="empty_b", test_cases=[])

    with pytest.raises(IncompatibleTestRunsError) as exc_info:
        compare_test_runs(
            baseline_run=empty_run_a,
            candidate_run=empty_run_b,
            judge_model=MockJudge(),
        )

    assert "must contain at least one test case" in str(exc_info.value)


def test_invalid_or_incomplete_judge_output():
    baseline_run = _create_sample_run("v1", ["A"])
    candidate_run = _create_sample_run("v2", ["B"])

    # Output that cannot be parsed as JSON or regex
    judge_model = MockJudge(
        verdicts=[
            "I cannot determine the result because this text has no JSON structure."
        ]
    )

    with pytest.raises(JudgeEvaluationError) as exc_info:
        compare_test_runs(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            judge_model=judge_model,
            async_mode=False,
        )

    assert "unparseable output" in str(exc_info.value).lower()


def test_raw_json_string_judge_output():
    baseline_run = _create_sample_run("v1", ["A"])
    candidate_run = _create_sample_run("v2", ["B"])

    # Output as valid JSON string with surrounding markdown
    raw_json = '```json\n{"winner": "candidate", "reason": "Candidate provided superior citations."}\n```'
    judge_model = MockJudge(verdicts=[raw_json])

    result = compare_test_runs(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "candidate"
    assert (
        result.comparisons[0].reason == "Candidate provided superior citations."
    )


def test_judge_model_failure():
    baseline_run = _create_sample_run("v1", ["A"])
    candidate_run = _create_sample_run("v2", ["B"])

    judge_model = MockJudge(should_fail=True)

    with pytest.raises(JudgeEvaluationError) as exc_info:
        compare_test_runs(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            judge_model=judge_model,
            async_mode=False,
        )

    assert "Judge model failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_compare_test_runs():
    baseline_run = _create_sample_run("v1", ["Out 1", "Out 2"])
    candidate_run = _create_sample_run("v2", ["Better 1", "Better 2"])

    judge_model = MockJudge(
        verdicts=[
            JudgeVerdict(winner="candidate", reason="Better quality 1"),
            JudgeVerdict(winner="baseline", reason="Better quality 2"),
        ]
    )

    result = await a_compare_test_runs(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
    )

    assert result.winner == "tie"
    assert result.candidate_wins == 1
    assert result.baseline_wins == 1
    assert len(result.comparisons) == 2


def test_file_path_and_dict_inputs(tmp_path: Path):
    baseline_run = _create_sample_run("v1", ["Base output"])
    candidate_run = _create_sample_run("v2", ["Cand output"])

    # Write baseline run to temporary JSON file
    baseline_path = tmp_path / "baseline_run.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_run.model_dump(by_alias=True), f)

    # Convert candidate run to dictionary
    candidate_dict = candidate_run.model_dump(by_alias=True)

    judge_model = MockJudge(
        verdicts=[JudgeVerdict(winner="candidate", reason="Cand output won.")]
    )

    result = compare_test_runs(
        baseline_run=str(baseline_path),
        candidate_run=candidate_dict,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "candidate"
    assert result.candidate_wins == 1
    assert result.baseline_run.identifier == "v1"
    assert result.candidate_run.identifier == "v2"


def test_judge_alias():
    baseline_run = _create_sample_run("v1", ["A"])
    candidate_run = _create_sample_run("v2", ["B"])

    judge_model = MockJudge(
        verdicts=[JudgeVerdict(winner="candidate", reason="Won.")]
    )

    # Calling via `judge(...)` alias
    result = judge(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        judge_model=judge_model,
        async_mode=False,
    )

    assert result.winner == "candidate"
