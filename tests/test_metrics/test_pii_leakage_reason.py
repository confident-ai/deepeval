import json

import pytest

from deepeval.metrics import PIILeakageMetric
from deepeval.test_case import LLMTestCase
from tests.test_metrics.system_one_fakes import ScriptedLLM


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("leaks_pii", [False, True])
def test_reason_prompt_explains_privacy_score(async_mode, leaks_pii):
    violation = "The response discloses a customer's email address."
    candidate = (
        "alex@example.com" if leaks_pii else "Thank you for reaching out."
    )
    replies = [json.dumps({"extracted_pii": [candidate] if leaks_pii else []})]
    if leaks_pii:
        replies.append(
            json.dumps({"verdicts": [{"verdict": "yes", "reason": violation}]})
        )
    replies.append(json.dumps({"reason": "Assessment complete."}))
    judge = ScriptedLLM(replies)
    metric = PIILeakageMetric(model=judge, async_mode=async_mode)

    score = metric.measure(
        LLMTestCase(input="Write a public reply.", actual_output=candidate),
        _show_indicator=False,
    )

    assert score == (0.0 if leaks_pii else 1.0)
    assert metric.reason == "Assessment complete."
    reason_prompt = judge.prompts[-1]
    assert f"Computed privacy score: {score:.2f}" in reason_prompt
    assert "Higher scores indicate fewer privacy violations" in reason_prompt
    assert "Treat the computed score as fixed" in reason_prompt
    assert (violation in reason_prompt) == leaks_pii
