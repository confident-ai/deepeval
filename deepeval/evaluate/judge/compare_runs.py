from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from deepeval.errors import IncompatibleTestRunsError, JudgeEvaluationError
from deepeval.evaluate.judge.schema import (
    JudgeVerdict,
    TestCaseComparison,
    TestRunComparisonResult,
    WinnerType,
)
from deepeval.evaluate.judge.template import JudgeLMTemplate
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_run.api import LLMApiTestCase
from deepeval.test_run.test_run import TestRun
from deepeval.utils import get_or_create_event_loop


def load_test_run(
    run: TestRun | str | Path | dict[str, Any],
) -> TestRun:
    """Load a TestRun from an object, JSON file path, or dictionary.

    Args:
        run: An instance of TestRun, a path to a test run JSON file, or a dict.

    Returns:
        A validated TestRun instance.

    Raises:
        TypeError: If the input is not a recognized type.
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If parsing the test run fails.
    """
    if isinstance(run, TestRun):
        return run

    if isinstance(run, (str, Path)):
        path = Path(run)
        if not path.is_file():
            raise FileNotFoundError(f"Test run file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(run, dict):
        data = run
    else:
        raise TypeError(
            f"Unsupported type for test run: {type(run)}. Expected TestRun, file path (str/Path), or dict."
        )

    # If wrapped under a top-level key (e.g., {"testRun": ...} or {"test_run": ...})
    for key in ("testRun", "test_run", "data"):
        if (
            isinstance(data, dict)
            and key in data
            and isinstance(data[key], dict)
        ):
            data = data[key]
            break

    if isinstance(data, dict):
        data = dict(data)
        if "test_cases" in data and "testCases" not in data:
            data["testCases"] = data["test_cases"]
        if (
            "conversational_test_cases" in data
            and "conversationalTestCases" not in data
        ):
            data["conversationalTestCases"] = data["conversational_test_cases"]

    try:
        return TestRun.model_validate(data)
    except AttributeError:
        # Fallback for Pydantic v1
        return TestRun(**data)


def validate_test_runs_compatibility(
    baseline_run: TestRun,
    candidate_run: TestRun,
) -> list[tuple[LLMApiTestCase, LLMApiTestCase]]:
    """Validate that two historical test runs are compatible and pair their test cases.

    Runs are compatible when:
      - Both runs contain at least one test case.
      - Both runs contain the exact same number of test cases.
      - Each test case in the baseline run matches a corresponding test case in the candidate run.

    Returns:
        A list of paired (baseline_test_case, candidate_test_case) tuples.

    Raises:
        IncompatibleTestRunsError: If the runs cannot be compared.
    """
    base_cases = baseline_run.test_cases or []
    cand_cases = candidate_run.test_cases or []

    if len(base_cases) == 0 or len(cand_cases) == 0:
        raise IncompatibleTestRunsError(
            f"Both test runs must contain at least one test case to compare. "
            f"Found {len(base_cases)} baseline cases and {len(cand_cases)} candidate cases."
        )

    if len(base_cases) != len(cand_cases):
        raise IncompatibleTestRunsError(
            f"Test case count mismatch: baseline run contains {len(base_cases)} cases, "
            f"while candidate run contains {len(cand_cases)} cases."
        )

    # First attempt: positional matching if inputs line up
    paired: list[tuple[LLMApiTestCase, LLMApiTestCase]] = []
    positional_match = True
    for i, (b, c) in enumerate(zip(base_cases, cand_cases)):
        if (b.input or "").strip() == (c.input or "").strip():
            paired.append((b, c))
        else:
            positional_match = False
            break

    if positional_match:
        return paired

    # Second attempt: match by exact input
    paired = []
    unmatched_candidate = list(cand_cases)
    for b in base_cases:
        b_input = (b.input or "").strip()
        match_idx = None
        for idx, c in enumerate(unmatched_candidate):
            if (c.input or "").strip() == b_input:
                match_idx = idx
                break

        if match_idx is None:
            raise IncompatibleTestRunsError(
                f"Test runs evaluate different test cases. Could not find a matching candidate "
                f"test case for baseline input: {b_input[:80]!r}"
            )
        c_match = unmatched_candidate.pop(match_idx)
        paired.append((b, c_match))

    return paired


def normalize_winner_verdict(raw_winner: Any) -> WinnerType:
    """Normalize free-form or schema winner tokens into 'baseline', 'candidate', or 'tie'."""
    if not raw_winner:
        return "tie"

    clean = str(raw_winner).strip().lower()
    clean = re.sub(r"[\"'\$]", "", clean).strip()

    if clean in {"candidate", "response b", "b", "model b"}:
        return "candidate"
    elif clean in {"baseline", "response a", "a", "model a"}:
        return "baseline"
    elif clean in {"tie", "equal", "draw", "none", "indeterminate", "same"}:
        return "tie"

    # Substring search fallback
    if "candidate" in clean or "response b" in clean:
        return "candidate"
    if "baseline" in clean or "response a" in clean:
        return "baseline"
    if "tie" in clean or "draw" in clean or "equal" in clean:
        return "tie"

    return "tie"


def _parse_judge_response(response: Any) -> JudgeVerdict:
    """Parse the judge model's raw output into a structured JudgeVerdict."""
    if isinstance(response, JudgeVerdict):
        winner = normalize_winner_verdict(response.winner)
        return JudgeVerdict(winner=winner, reason=response.reason)

    if isinstance(response, dict):
        winner = normalize_winner_verdict(response.get("winner"))
        reason = str(response.get("reason", "No reason provided."))
        return JudgeVerdict(winner=winner, reason=reason)

    if not isinstance(response, str):
        raise JudgeEvaluationError(
            f"Unexpected response type from judge model: {type(response)}"
        )

    try:
        data = trimAndLoadJson(response)
        winner = normalize_winner_verdict(data.get("winner"))
        reason = str(data.get("reason", "No reason provided."))
        return JudgeVerdict(winner=winner, reason=reason)
    except Exception as e:
        # Fallback regex extraction if trimAndLoadJson fails
        winner_match = re.search(
            r'"winner"\s*:\s*"([^"]+)"', response, re.IGNORECASE
        )
        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]+)"', response, re.IGNORECASE
        )

        if winner_match:
            winner = normalize_winner_verdict(winner_match.group(1))
            reason = (
                reason_match.group(1)
                if reason_match
                else "Reason extracted via regex fallback."
            )
            return JudgeVerdict(winner=winner, reason=reason)

        raise JudgeEvaluationError(
            f"Judge model returned unparseable output: {response!r}"
        ) from e


def _aggregate_results(
    baseline_run: TestRun,
    candidate_run: TestRun,
    comparisons: list[TestCaseComparison],
) -> TestRunComparisonResult:
    """Aggregate individual case comparisons into an overall TestRunComparisonResult."""
    baseline_wins = sum(1 for c in comparisons if c.winner == "baseline")
    candidate_wins = sum(1 for c in comparisons if c.winner == "candidate")
    ties = sum(1 for c in comparisons if c.winner == "tie")
    total = len(comparisons)

    overall_winner: WinnerType
    if candidate_wins > baseline_wins:
        overall_winner = "candidate"
        reason = (
            f"Candidate won {candidate_wins} out of {total} test cases "
            f"(Baseline won {baseline_wins}, {ties} ties)."
        )
    elif baseline_wins > candidate_wins:
        overall_winner = "baseline"
        reason = (
            f"Baseline won {baseline_wins} out of {total} test cases "
            f"(Candidate won {candidate_wins}, {ties} ties)."
        )
    else:
        overall_winner = "tie"
        reason = (
            f"The runs tied with {baseline_wins} wins each "
            f"out of {total} test cases ({ties} ties)."
        )

    return TestRunComparisonResult(
        winner=overall_winner,
        reason=reason,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        comparisons=comparisons,
        baseline_wins=baseline_wins,
        candidate_wins=candidate_wins,
        ties=ties,
    )


def compare_test_runs(
    baseline_run: TestRun | str | Path | dict[str, Any],
    candidate_run: TestRun | str | Path | dict[str, Any],
    judge_model: str | DeepEvalBaseLLM | None = None,
    criteria: str | None = None,
    async_mode: bool = True,
) -> TestRunComparisonResult:
    """Compare two historical DeepEval test runs using a configurable LLM judge.

    Args:
        baseline_run: Baseline test run (TestRun instance, JSON file path, or dict).
        candidate_run: Candidate test run (TestRun instance, JSON file path, or dict).
        judge_model: The LLM judge (model name string, DeepEvalBaseLLM instance, or None for default).
        criteria: Custom evaluation criteria guiding the judge's pairwise assessment.
        async_mode: If True, execute evaluations asynchronously via event loop.

    Returns:
        A structured TestRunComparisonResult detailing the overall winner, reason, and case-by-case comparisons.

    Raises:
        IncompatibleTestRunsError: If test runs differ in size or inputs.
        JudgeEvaluationError: If the judge model fails or returns unparseable outputs.
    """
    if async_mode:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            a_compare_test_runs(
                baseline_run=baseline_run,
                candidate_run=candidate_run,
                judge_model=judge_model,
                criteria=criteria,
            )
        )

    base = load_test_run(baseline_run)
    cand = load_test_run(candidate_run)
    pairs = validate_test_runs_compatibility(base, cand)
    model, _ = initialize_model(judge_model)

    comparisons: list[TestCaseComparison] = []
    for base_case, cand_case in pairs:
        prompt = JudgeLMTemplate.compare_outputs(
            input_text=base_case.input,
            baseline_output=base_case.actual_output,
            candidate_output=cand_case.actual_output,
            expected_output=base_case.expected_output
            or cand_case.expected_output,
            context=base_case.context or cand_case.context,
            criteria=criteria,
        )

        try:
            raw_res = model.generate_with_schema(prompt, schema=JudgeVerdict)
            verdict = _parse_judge_response(raw_res)
        except Exception as e:
            if isinstance(e, JudgeEvaluationError):
                raise
            raise JudgeEvaluationError(
                f"Judge model failed during comparison: {e}"
            ) from e

        comparisons.append(
            TestCaseComparison(
                name=base_case.name or cand_case.name,
                input=base_case.input,
                baseline_output=base_case.actual_output,
                candidate_output=cand_case.actual_output,
                expected_output=base_case.expected_output
                or cand_case.expected_output,
                context=base_case.context or cand_case.context,
                winner=verdict.winner,
                reason=verdict.reason,
            )
        )

    return _aggregate_results(base, cand, comparisons)


async def a_compare_test_runs(
    baseline_run: TestRun | str | Path | dict[str, Any],
    candidate_run: TestRun | str | Path | dict[str, Any],
    judge_model: str | DeepEvalBaseLLM | None = None,
    criteria: str | None = None,
) -> TestRunComparisonResult:
    """Asynchronously compare two historical DeepEval test runs using a configurable LLM judge."""
    import asyncio

    base = load_test_run(baseline_run)
    cand = load_test_run(candidate_run)
    pairs = validate_test_runs_compatibility(base, cand)
    model, _ = initialize_model(judge_model)

    async def _evaluate_pair(
        base_case: LLMApiTestCase, cand_case: LLMApiTestCase
    ) -> TestCaseComparison:
        prompt = JudgeLMTemplate.compare_outputs(
            input_text=base_case.input,
            baseline_output=base_case.actual_output,
            candidate_output=cand_case.actual_output,
            expected_output=base_case.expected_output
            or cand_case.expected_output,
            context=base_case.context or cand_case.context,
            criteria=criteria,
        )

        try:
            raw_res = await model.a_generate_with_schema(
                prompt, schema=JudgeVerdict
            )
            verdict = _parse_judge_response(raw_res)
        except Exception as e:
            if isinstance(e, JudgeEvaluationError):
                raise
            raise JudgeEvaluationError(
                f"Judge model failed during comparison: {e}"
            ) from e

        return TestCaseComparison(
            name=base_case.name or cand_case.name,
            input=base_case.input,
            baseline_output=base_case.actual_output,
            candidate_output=cand_case.actual_output,
            expected_output=base_case.expected_output
            or cand_case.expected_output,
            context=base_case.context or cand_case.context,
            winner=verdict.winner,
            reason=verdict.reason,
        )

    tasks = [_evaluate_pair(b, c) for b, c in pairs]
    comparisons = await asyncio.gather(*tasks)

    return _aggregate_results(base, cand, list(comparisons))


# Convenient alias matching issue #262 proposal
judge = compare_test_runs
