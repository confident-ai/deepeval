# tests/test_core/test_optimization/test_mutations/test_prompt_rewriter.py
from dataclasses import dataclass
import random

import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimization.mutations.prompt_rewriter import (
    _compose_prompt_messages,
    _normalize_llm_output_to_text,
    PromptRewriter,
    MetricAwareLLMRewriter,
)
from deepeval.optimization.configs import (
    PromptListMutationConfig,
    PromptListMutationTargetType,
)
from deepeval.prompt.api import PromptMessage
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


def test_prompt_rewriter_list_default_mutates_random_message():
    """
    With default PromptListMutationConfig (RANDOM, no role),
    exactly one message in the list is rewritten and roles are preserved.
    """
    rewriter = PromptRewriter()
    old = Prompt(
        messages_template=[
            PromptMessage(role="user", content="m0"),
            PromptMessage(role="system", content="m1"),
            PromptMessage(role="assistant", content="m2"),
        ]
    )

    def model_callback(**kwargs):
        return "rewritten list"

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text="feedback",
        model_callback=model_callback,
    )

    assert isinstance(new, Prompt)
    assert new.messages_template is not None

    old_msgs = old.messages_template
    new_msgs = new.messages_template

    # Exactly one message’s content should change.
    changed_indices = [
        i
        for i, (old_msg, new_msg) in enumerate(zip(old_msgs, new_msgs))
        if old_msg.content != new_msg.content
    ]

    assert changed_indices, "At least one message should be rewritten"
    assert len(changed_indices) == 1

    # Roles should be preserved for all messages.
    for old_msg, new_msg in zip(old_msgs, new_msgs):
        assert old_msg.role == new_msg.role


def test_prompt_rewriter_list_random_uses_random_state_and_role_filter():
    """
    RANDOM mode with a seeded RNG + target_role should pick among the
    matching-role messages only, deterministically.
    """
    cfg = PromptListMutationConfig(
        target_type=PromptListMutationTargetType.RANDOM,
        target_role="assistant",
    )
    rewriter = PromptRewriter(
        list_mutation_config=cfg,
        random_state=random.Random(0),
    )
    old = Prompt(
        messages_template=[
            PromptMessage(role="system", content="sys"),
            PromptMessage(role="assistant", content="a0"),
            PromptMessage(role="assistant", content="a1"),
        ]
    )

    def model_callback(**kwargs):
        return "rewrite"

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text="feedback",
        model_callback=model_callback,
    )

    msgs = new.messages_template
    # System message stays unchanged.
    assert msgs[0].content == "sys"

    # With Random(0) and candidates [1, 2], we expect index 2.
    assert msgs[1].content == "a0"
    assert msgs[2].content == "rewrite"


@pytest.mark.parametrize("target_index", [0, 1, 2])
def test_prompt_rewriter_list_fixed_index_mutates_selected_message(
    target_index: int,
):
    """
    When target_type == FIXED_INDEX, the message at target_index is rewritten
    and all other messages remain unchanged. Roles are preserved.
    """
    cfg = PromptListMutationConfig(
        target_type=PromptListMutationTargetType.FIXED_INDEX,
        target_index=target_index,
    )
    rewriter = PromptRewriter(list_mutation_config=cfg)

    old = Prompt(
        messages_template=[
            PromptMessage(role="user", content="m0"),
            PromptMessage(role="system", content="m1"),
            PromptMessage(role="assistant", content="m2"),
        ]
    )

    def model_callback(**kwargs):
        # Mark which index we expected to rewrite to make debugging easier
        return f"rewritten-{target_index}"

    new = rewriter.rewrite(
        module_id="__module__",
        old_prompt=old,
        feedback_text="feedback",
        model_callback=model_callback,
    )

    assert isinstance(new, Prompt)
    assert new.messages_template is not None

    old_msgs = old.messages_template
    new_msgs = new.messages_template

    for idx, (old_msg, new_msg) in enumerate(zip(old_msgs, new_msgs)):
        # Roles are always preserved
        assert new_msg.role == old_msg.role

        if idx == target_index:
            # Only the selected index is rewritten
            assert new_msg.content == f"rewritten-{target_index}"
        else:
            assert new_msg.content == old_msg.content


def test_prompt_rewriter_fixed_index_out_of_range_raises():
    cfg = PromptListMutationConfig(
        target_type=PromptListMutationTargetType.FIXED_INDEX,
        target_index=10,
    )
    rewriter = PromptRewriter(list_mutation_config=cfg)
    old = Prompt(messages_template=[PromptMessage(role="user", content="m0")])

    def model_callback(**kwargs):
        return "rewrite"

    with pytest.raises(DeepEvalError):
        rewriter.rewrite(
            module_id="__module__",
            old_prompt=old,
            feedback_text="feedback",
            model_callback=model_callback,
        )


def test_prompt_rewriter_accepts_int_random_seed_and_converts_to_random():
    """
    Passing an int seed for random_state should be accepted and
    converted into a random.Random instance.
    """
    rewriter = PromptRewriter(random_state=123)
    assert isinstance(rewriter.random_state, random.Random)


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
        max_chars=4000,
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
        max_chars=4000,
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
