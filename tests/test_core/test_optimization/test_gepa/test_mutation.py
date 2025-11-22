from dataclasses import dataclass

import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimization.gepa.mutation import (
    _compose_prompt_messages,
    _normalize_llm_output_to_text,
    PromptRewriter,
    MetricAwareLLMRewriter,
)
from deepeval.prompt.prompt import Prompt


########################
# Helper test doubles  #
########################


@dataclass
class StubMetricInfo:
    """
    Minimal stand-in for MetricInfo, using only the attributes
    MetricAwareLLMRewriter actually accesses.
    """

    name: str
    rubric: str | None = None


###############################
# _compose_prompt_messages    #
###############################


@pytest.mark.parametrize(
    "system,user,expected",
    [
        ("System", "User", "System\n\nUser"),
        ("", "User only", "User only"),
        ("   ", "  Trim me  ", "Trim me"),
    ],
)
def test_compose_prompt_messages(system, user, expected):
    assert _compose_prompt_messages(system, user) == expected


########################################
# _normalize_llm_output_to_text        #
########################################


def test_normalize_llm_output_to_text_str_and_tuple():
    assert _normalize_llm_output_to_text("  hi  ") == "hi"

    text = _normalize_llm_output_to_text(("  hello ", 1.23))
    assert text == "hello"


def test_normalize_llm_output_to_text_dict_json_and_unicode():
    data = {"answer": "café", "score": 0.9}
    out = _normalize_llm_output_to_text(data)

    # Should be JSON and preserve unicode characters
    assert '"answer"' in out
    assert "café" in out
    assert '"score"' in out


def test_normalize_llm_output_to_text_unserializable_falls_back_to_str():
    class Unserializable:
        def __repr__(self):
            return "<UnserializableObject>"

    obj = Unserializable()
    out = _normalize_llm_output_to_text(obj)

    assert isinstance(out, str)
    assert "UnserializableObject" in out


#########################
# PromptRewriter (sync) #
#########################


def test_prompt_rewriter_requires_model_callback():
    rewriter = PromptRewriter()
    old = Prompt(text_template="Original prompt")

    # Passing model_callback=None should be rejected by validate_callback
    with pytest.raises(DeepEvalError):
        rewriter.rewrite(
            module_id="__module__",
            old_prompt=old,
            feedback_text="some feedback",
            model_callback=None,
        )


def test_prompt_rewriter_returns_old_prompt_when_feedback_empty():
    rewriter = PromptRewriter()
    old = Prompt(text_template="Keep me")

    called = {"count": 0}

    def model_callback(**kwargs):
        called["count"] += 1
        return "should-not-be-used"

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text="   ",  # whitespace only
        model_callback=model_callback,
    )

    # No rewrite when feedback is empty or whitespace
    assert new is old
    assert called["count"] == 0  # callback was never used


def test_prompt_rewriter_uses_model_callback_and_merges_context():
    rewriter = PromptRewriter()
    old = Prompt(text_template="Original prompt text")
    feedback = "Please be more specific."

    captured = {}

    def model_callback(
        hook: str,
        prompt_text: str | None = None,
        feedback_text: str | None = None,
        **kwargs,
    ):
        captured["hook"] = hook
        captured["prompt_text"] = prompt_text
        captured["feedback_text"] = feedback_text
        captured["kwargs"] = kwargs
        return "rewritten prompt"

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text=feedback,
        model_callback=model_callback,
    )

    # Callback must have been used and its text should become the new prompt
    assert isinstance(new, Prompt)
    assert new.text_template == "rewritten prompt"

    # Callback should get the right hook and merged prompt text
    assert captured["hook"] == "prompt_rewrite"
    prompt_text = captured["prompt_text"] or ""
    feedback_text = captured["feedback_text"]

    assert "Original prompt text" in prompt_text
    assert "Please be more specific." in prompt_text
    assert feedback_text == feedback


def test_prompt_rewriter_returns_old_prompt_if_new_text_empty_after_trim():
    rewriter = PromptRewriter()
    old = Prompt(text_template="Original prompt")
    feedback = "Try to improve this."

    def model_callback(**kwargs):
        # Return only whitespace, this will become empty after trimming
        return "   "

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text=feedback,
        model_callback=model_callback,
    )

    # Since the rewrite produced an empty string, we keep the old prompt
    assert new is old


##########################
# PromptRewriter (async) #
##########################


@pytest.mark.asyncio
async def test_prompt_rewriter_async_uses_model_callback():
    rewriter = PromptRewriter()
    old = Prompt(text_template="Original async prompt")
    feedback = "Async feedback."

    captured: dict = {}

    async def model_callback(
        hook: str,
        prompt_text: str | None = None,
        feedback_text: str | None = None,
        prompt: Prompt | None = None,
        **kwargs,
    ):
        # This mirrors what build_model_callback_kwargs / a_invoke_model_callback send:
        #   - prompt
        #   - prompt_text
        #   - feedback_text
        #   - hook (injected by a_invoke_model_callback)
        captured["hook"] = hook
        captured["prompt_text"] = prompt_text
        captured["feedback_text"] = feedback_text
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "async callback rewrite"

    new = await rewriter.a_rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text=feedback,
        model_callback=model_callback,
    )

    # Callback must have been used and its text should become the new prompt
    assert isinstance(new, Prompt)
    assert new.text_template == "async callback rewrite"

    # Sanity-check that merged prompt+feedback made it into the callback
    assert captured["hook"] == "prompt_rewrite"
    assert captured["prompt"] is old

    prompt_text = captured.get("prompt_text") or ""
    feedback_text = captured.get("feedback_text")

    assert "Original async prompt" in prompt_text
    assert "Async feedback." in prompt_text
    assert feedback_text == feedback


#####################################
# MetricAwareLLMRewriter._compose   #
#####################################


def test_metric_aware_rewriter_compose_messages_without_metrics():
    rewriter = MetricAwareLLMRewriter(metrics_info=None)
    prompt = Prompt(text_template="Base prompt")
    feedback = "Some issues found."

    system, user = rewriter._compose_messages(
        module_id="__module__",
        old_prompt=prompt,
        feedback_text=feedback,
    )

    assert "multi-step LLM pipeline" in system
    assert "[Module]" in user
    assert "__module__" in user
    assert "[Current Prompt]" in user
    assert "Base prompt" in user
    assert "[Feedback]" in user
    assert "Some issues found." in user
    # No explicit [Metric Rubrics] block when metrics_info is empty/None
    assert "[Metric Rubrics]" not in user


def test_metric_aware_rewriter_compose_messages_with_rubrics_and_defaults():
    metrics_info = [
        StubMetricInfo(
            name="AnswerRelevancy",
            rubric="Focus on relevant answers.",
        ),
        StubMetricInfo(
            name="Toxicity",
            rubric="",  # empty rubric -> default text
        ),
    ]
    rewriter = MetricAwareLLMRewriter(
        metrics_info=metrics_info,
        max_metrics_in_prompt=10,
    )

    prompt = Prompt(text_template="Base prompt")
    feedback = "Some issues found."

    system, user = rewriter._compose_messages(
        module_id="__module__",
        old_prompt=prompt,
        feedback_text=feedback,
    )

    # System message still describes metric-aware refinement
    assert "metric rubrics" in system

    # Rubric block should include both metrics, with custom + default text
    assert "[Metric Rubrics]" in user
    assert "- AnswerRelevancy: Focus on relevant answers." in user
    assert "- Toxicity: Optimize for this metric’s quality criteria." in user


def test_metric_aware_rewriter_respects_max_metrics_in_prompt():
    metrics_info = [
        StubMetricInfo(name=f"Metric{i}", rubric=f"Rubric {i}")
        for i in range(10)
    ]
    rewriter = MetricAwareLLMRewriter(
        metrics_info=metrics_info,
        max_metrics_in_prompt=3,
    )

    prompt = Prompt(text_template="Base prompt")
    feedback = "Some issues found."

    _, user = rewriter._compose_messages(
        module_id="__module__",
        old_prompt=prompt,
        feedback_text=feedback,
    )

    # Only the first 3 metrics should appear
    for i in range(3):
        assert f"- Metric{i}: Rubric {i}" in user

    # Later metrics must not be included
    for i in range(3, 10):
        assert f"Metric{i}" not in user
