import pytest
from unittest.mock import MagicMock, patch
from deepeval.metrics.process_integrity import ProcessIntegrityMetric
from deepeval.test_case import LLMTestCase


def make_test_case(steps):
    tc = LLMTestCase(input="test", actual_output="test")
    tc.steps = steps
    return tc


def make_metric_with_mock_model(responses: list):
    """
    Returns a ProcessIntegrityMetric with a mock model that returns
    responses in order, one per _call_model invocation.
    """
    metric = ProcessIntegrityMetric(threshold=0.5)
    mock_model = MagicMock()
    mock_model.generate.side_effect = [(r, None) for r in responses]
    metric.model = mock_model
    return metric


# ── Test 1: clean trajectory — expect score > 0.5 ───────────────────────────


def test_clean_trajectory():
    steps = [
        "Searched for flights. Conclusion: cheapest option is United at $320.",
        "Checked baggage fees for United. Conclusion: total cost is $370.",
        "Compared with Delta at $350 all-in. Conclusion: Delta is cheaper overall.",
        "Recommended Delta to user.",
    ]
    # Pass 1 responses (extract_conclusion x4)
    p1 = [
        '{"conclusion": "Cheapest option is United at $320.", "is_substantive": true}',
        '{"conclusion": "Total cost with United is $370.", "is_substantive": true}',
        '{"conclusion": "Delta is cheaper overall at $350 all-in.", "is_substantive": true}',
        '{"conclusion": "Delta recommended to user.", "is_substantive": true}',
    ]
    # Pass 2 responses (check_integrity x3 — step 0 auto-PASS, steps 1, 2, and 3 checked)
    p2 = [
        '{"verdict": "PASS", "reason": "Total cost finding is consistent with cheapest flight baseline."}',
        '{"verdict": "PASS", "reason": "Consistent with prior cost analysis."}',
        '{"verdict": "PASS", "reason": "Recommendation follows from cheapest option conclusion."}',
    ]
    metric = make_metric_with_mock_model(p1 + p2)
    tc = make_test_case(steps)
    score = metric.measure(tc)
    assert score > 0.5
    assert metric.success is True
    assert all(
        k in metric.score_breakdown
        for k in ["step_0", "step_1", "step_2", "step_3"]
    )


# ── Test 2: contradicting trajectory — expect score < 0.5 ───────────────────


def test_contradicting_trajectory():
    steps = [
        "Analyzed data. Conclusion: Model A outperforms Model B.",
        "Ran second analysis. Conclusion: Model B outperforms Model A.",
        "Selected Model A as final recommendation.",
    ]
    p1 = [
        '{"conclusion": "Model A outperforms Model B.", "is_substantive": true}',
        '{"conclusion": "Model B outperforms Model A.", "is_substantive": true}',
        '{"conclusion": "Model A selected as final recommendation.", "is_substantive": true}',
    ]
    # Step 0 auto-PASS; steps 1 and 2 checked
    p2 = [
        '{"verdict": "FAIL", "reason": "Directly contradicts step 0 conclusion that Model A outperforms Model B.", "severity": "high"}',
        '{"verdict": "FAIL", "reason": "Recommendation contradicts step 1 conclusion without reconciliation.", "severity": "high"}',
    ]
    metric = make_metric_with_mock_model(p1 + p2)
    tc = make_test_case(steps)
    score = metric.measure(tc)
    assert score < 0.5
    assert metric.success is False
    assert metric.score_breakdown["step_1"]["verdict"] == "FAIL"
    assert metric.score_breakdown["step_2"]["verdict"] == "FAIL"


# ── Test 3: all tool calls — expect all SKIP, score = 0.0 ───────────────────


def test_all_tool_calls():
    steps = [
        "Called search_api(query='flights')",
        "Called get_price(flight_id=123)",
        "Called book_flight(flight_id=123)",
    ]
    p1 = [
        '{"conclusion": "", "is_substantive": false}',
        '{"conclusion": "", "is_substantive": false}',
        '{"conclusion": "", "is_substantive": false}',
    ]
    # No Pass 2 calls — all steps SKIP
    metric = make_metric_with_mock_model(p1)
    tc = make_test_case(steps)
    score = metric.measure(tc)
    assert score == 0.0
    assert all(
        metric.score_breakdown[f"step_{i}"]["verdict"] == "SKIP"
        for i in range(3)
    )
