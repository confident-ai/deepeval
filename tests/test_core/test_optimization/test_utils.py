import asyncio
import random

import pytest

from tests.test_core.stubs import StubProvider, StubModelSettings, StubPrompt
from deepeval.prompt.prompt import Prompt, PromptMessage
from deepeval.errors import DeepEvalError
from deepeval.optimization.types import PromptConfiguration, OptimizationReport
from deepeval.optimization.utils import (
    a_invoke_model_callback,
    build_model_callback_kwargs,
    generate_module_id,
    invoke_model_callback,
    normalize_seed_prompts,
    split_goldens,
    build_prompt_config_snapshots,
    inflate_prompts_from_report,
    validate_callback,
    validate_instance,
    validate_sequence_of,
)

#################
# split_goldens #
#################


def test_split_goldens_raises_for_negative_pareto_size() -> None:
    goldens = ["g0", "g1"]
    rng = random.Random(0)

    with pytest.raises(ValueError, match="pareto_size must be >= 0"):
        split_goldens(goldens, -1, random_state=rng)


def test_split_goldens_empty_returns_empty_pairs() -> None:
    goldens: list[str] = []
    rng = random.Random(0)

    d_feedback, d_pareto = split_goldens(
        goldens, pareto_size=3, random_state=rng
    )

    assert d_feedback == []
    assert d_pareto == []


def test_split_goldens_single_all_goes_to_pareto() -> None:
    goldens = ["g0"]
    rng = random.Random(0)

    d_feedback, d_pareto = split_goldens(
        goldens, pareto_size=3, random_state=rng
    )

    # With a single example, we can’t form a feedback set;
    # everything goes to D_pareto.
    assert d_feedback == []
    assert d_pareto == ["g0"]


def test_split_goldens_zero_pareto_uses_all_for_feedback() -> None:
    goldens = ["g0", "g1", "g2"]
    rng = random.Random(0)

    d_feedback, d_pareto = split_goldens(goldens, 0, random_state=rng)

    assert d_pareto == []
    # feedback keeps original order
    assert d_feedback == goldens


def test_split_goldens_large_pareto_leaves_at_least_one_feedback() -> None:
    goldens = ["g0", "g1", "g2", "g3"]
    rng = random.Random(0)

    d_feedback, d_pareto = split_goldens(goldens, 10, random_state=rng)

    # We always keep at least one example for D_feedback when total >= 2
    assert len(d_pareto) == 3
    assert len(d_feedback) == 1

    # Disjoint and covering the whole set
    assert set(d_feedback).isdisjoint(d_pareto)
    combined = d_feedback + d_pareto
    assert sorted(combined, key=lambda g: goldens.index(g)) == goldens


def test_split_goldens_deterministic_and_disjoint_with_fixed_seed() -> None:
    goldens = [f"g{i}" for i in range(10)]

    rng1 = random.Random(1234)
    d_feedback1, d_pareto1 = split_goldens(
        goldens, pareto_size=3, random_state=rng1
    )

    rng2 = random.Random(1234)
    d_feedback2, d_pareto2 = split_goldens(
        goldens, pareto_size=3, random_state=rng2
    )

    # determinism
    assert d_feedback1 == d_feedback2
    assert d_pareto1 == d_pareto2

    # correct sizes
    assert len(d_pareto1) == 3
    assert len(d_feedback1) == len(goldens) - 3

    # disjoint and covering the whole set
    assert set(d_feedback1).isdisjoint(d_pareto1)
    combined = d_feedback1 + d_pareto1
    assert sorted(combined, key=lambda g: goldens.index(g)) == goldens


###############################################
# generate_module_id / normalize_seed_prompts #
###############################################


def test_generate_module_id_includes_alias_label_and_model_info() -> None:
    existing: set[str] = set()
    prompt = StubPrompt(
        alias="My Prompt",
        label="For Chatbot",
        model_settings=StubModelSettings(
            provider=StubProvider("OPEN_AI"),
            name="gpt-4o-mini",
        ),
    )

    module_id = generate_module_id(prompt, index=0, existing=existing)

    assert module_id == "my-prompt-for-chatbot-open-ai-gpt-4o-mini"
    assert module_id in existing


def test_generate_module_id_uses_fallback_and_dedupes() -> None:
    existing: set[str] = set()

    p1 = StubPrompt()
    id1 = generate_module_id(p1, index=0, existing=existing)
    assert id1 == "module-1"

    # Same parameters and index but different prompt should get a suffixed id
    p2 = StubPrompt()
    id2 = generate_module_id(p2, index=0, existing=existing)
    assert id2 == "module-1-2"

    assert id1 in existing
    assert id2 in existing
    assert id1 != id2


def test_generate_module_id_truncates_long_ids_to_64_chars() -> None:
    long_alias = "A" * 100
    existing: set[str] = set()
    prompt = StubPrompt(alias=long_alias)

    module_id = generate_module_id(prompt, index=0, existing=existing)

    assert len(module_id) <= 64
    # base should not be empty
    assert module_id != ""


def test_normalize_seed_prompts_returns_shallow_copy_for_dict() -> None:
    prompt1 = StubPrompt(alias="A")
    prompt2 = StubPrompt(alias="B")
    seed = {"m1": prompt1, "m2": prompt2}

    normalized = normalize_seed_prompts(seed)

    assert normalized is not seed
    assert normalized == seed
    # Values are the same objects (shallow copy)
    assert normalized["m1"] is prompt1
    assert normalized["m2"] is prompt2


def test_normalize_seed_prompts_generates_unique_ids_for_list() -> None:
    p1 = StubPrompt(alias="First Prompt")
    p2 = StubPrompt(alias="Second Prompt")
    prompts = [p1, p2]

    normalized = normalize_seed_prompts(prompts)

    # Values preserved
    assert set(normalized.values()) == {p1, p2}
    # Unique, string keys generated
    keys = list(normalized.keys())
    assert all(isinstance(k, str) for k in keys)
    assert len(keys) == len(set(keys)) == len(prompts)


#####################
# validate_callback #
#####################


def test_validate_callback_raises_when_missing() -> None:
    with pytest.raises(DeepEvalError) as excinfo:
        validate_callback(
            component="PromptOptimizer",
            model_callback=None,
        )

    msg = str(excinfo.value)
    assert "requires a `model_callback`" in msg


def test_validate_callback_returns_same_callable() -> None:
    def cb(**_kwargs):
        return "ok"

    result = validate_callback(
        component="PromptOptimizer",
        model_callback=cb,
    )

    assert result is cb


###############################
# build_model_callback_kwargs #
###############################


def test_build_model_callback_kwargs_populates_all_fields() -> None:
    prompt = StubPrompt(alias="alias")
    golden = object()
    feedback_text = "feedback"
    prompt_text = "pt"
    prompt_messages = ["m1", "m2"]

    kwargs = build_model_callback_kwargs(
        prompt=prompt,
        prompt_text=prompt_text,
        prompt_messages=prompt_messages,
        golden=golden,
        feedback_text=feedback_text,
    )

    assert kwargs["prompt"] is prompt
    assert kwargs["prompt_text"] == prompt_text
    assert kwargs["prompt_messages"] == prompt_messages
    assert kwargs["golden"] is golden
    assert kwargs["feedback_text"] == feedback_text

    assert set(kwargs.keys()) == {
        "prompt",
        "prompt_text",
        "prompt_messages",
        "golden",
        "feedback_text",
    }


def test_build_model_callback_kwargs_defaults_missing_fields_to_none() -> None:
    kwargs = build_model_callback_kwargs()

    assert kwargs["prompt"] is None
    assert kwargs["prompt_text"] is None
    assert kwargs["prompt_messages"] is None
    assert kwargs["golden"] is None
    assert kwargs["feedback_text"] is None


###################################################
# invoke_model_callback / a_invoke_model_callback #
###################################################


def test_invoke_model_callback_filters_kwargs_and_injects_hook() -> None:
    captured: dict[str, str] = {}

    def cb(prompt_text, hook):
        captured["prompt_text"] = prompt_text
        captured["hook"] = hook
        return "ok"

    candidate_kwargs = {
        "prompt": "ignored",
        "prompt_text": "the prompt",
        "golden": "ignored",
        "feedback_text": "also ignored",
    }

    result = invoke_model_callback(
        hook="score_generate",
        model_callback=cb,
        candidate_kwargs=candidate_kwargs,
    )

    assert result == "ok"
    assert captured == {
        "prompt_text": "the prompt",
        "hook": "score_generate",
    }


def test_invoke_model_callback_does_not_pass_unsupported_kwargs() -> None:
    captured = {}

    def cb(prompt_text):
        captured["prompt_text"] = prompt_text
        return "ok"

    candidate_kwargs = {
        "prompt": "ignored",
        "prompt_text": "the prompt",
        "golden": "ignored",
        "feedback_text": "ignored",
    }

    result = invoke_model_callback(
        hook="score_generate",
        model_callback=cb,
        candidate_kwargs=candidate_kwargs,
    )

    assert result == "ok"
    assert captured == {"prompt_text": "the prompt"}


def test_invoke_model_callback_raises_on_awaitable_returned_in_sync_context() -> (
    None
):
    async def async_cb(prompt_text):
        return f"async:{prompt_text}"

    candidate_kwargs = {"prompt_text": "the prompt"}

    with pytest.raises(DeepEvalError) as excinfo:
        invoke_model_callback(
            hook="score_generate",
            model_callback=async_cb,
            candidate_kwargs=candidate_kwargs,
        )

    msg = str(excinfo.value)
    assert "returned an awaitable from a synchronous context" in msg


def test_a_invoke_model_callback_supports_async_callback_and_injects_hook() -> (
    None
):
    captured: dict[str, str] = {}

    async def async_cb(prompt_text, hook):
        captured["prompt_text"] = prompt_text
        captured["hook"] = hook
        return "ok"

    candidate_kwargs = {
        "prompt": "ignored",
        "prompt_text": "the prompt",
        "golden": "ignored",
        "feedback_text": "ignored",
    }

    async def runner():
        result = await a_invoke_model_callback(
            hook="score_generate",
            model_callback=async_cb,
            candidate_kwargs=candidate_kwargs,
        )
        assert result == "ok"

    asyncio.run(runner())

    assert captured == {
        "prompt_text": "the prompt",
        "hook": "score_generate",
    }


def test_a_invoke_model_callback_supports_sync_callback() -> None:
    captured: dict[str, str] = {}

    def cb(prompt_text, hook):
        captured["prompt_text"] = prompt_text
        captured["hook"] = hook
        return "ok"

    candidate_kwargs = {
        "prompt": "ignored",
        "prompt_text": "the prompt",
        "golden": "ignored",
        "feedback_text": "ignored",
    }

    async def runner():
        result = await a_invoke_model_callback(
            hook="score_generate",
            model_callback=cb,
            candidate_kwargs=candidate_kwargs,
        )
        assert result == "ok"

    asyncio.run(runner())

    assert captured == {
        "prompt_text": "the prompt",
        "hook": "score_generate",
    }


################
# report utils #
################


def test_build_prompt_config_snapshots_includes_text_and_list_prompts() -> None:
    """
    build_prompt_config_snapshots should:
      - include all configurations by id,
      - preserve parent relationships,
      - produce TEXT and LIST snapshots with expected shape.
    """
    # Root configuration with one TEXT and one LIST prompt
    root_cfg = PromptConfiguration.new(
        prompts={
            "text-module": Prompt(text_template="Hello, world"),
            "list-module": Prompt(
                messages_template=[
                    PromptMessage(role="system", content="sys"),
                    PromptMessage(role="user", content="hi"),
                ]
            ),
        }
    )

    # Child configuration with a different TEXT prompt
    child_cfg = PromptConfiguration.new(
        prompts={
            "text-module": Prompt(text_template="Child prompt"),
        },
        parent=root_cfg.id,
    )

    snapshots = build_prompt_config_snapshots(
        {root_cfg.id: root_cfg, child_cfg.id: child_cfg}
    )

    # We should have both configuration ids
    assert set(snapshots.keys()) == {root_cfg.id, child_cfg.id}

    root_snap = snapshots[root_cfg.id]
    child_snap = snapshots[child_cfg.id]

    # Parent relationships
    assert root_snap["parent"] is None
    assert child_snap["parent"] == root_cfg.id

    # TEXT module snapshot
    text_mod = root_snap["prompts"]["text-module"]
    assert text_mod["type"] == "TEXT"
    assert text_mod["text_template"] == "Hello, world"

    # LIST module snapshot
    list_mod = root_snap["prompts"]["list-module"]
    assert list_mod["type"] == "LIST"
    assert list_mod["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_inflate_prompts_from_report_reconstructs_prompts() -> None:
    """
    inflate_prompts_from_report should turn snapshot dicts into real Prompt objects
    for both TEXT and LIST modules.
    """
    runtime = {
        "optimization_id": "opt-123",
        "best_id": "cfg-best",
        "accepted_iterations": [],
        "pareto_scores": {"cfg-best": [1.0]},
        "parents": {"cfg-best": None},
        "prompt_configurations": {
            "cfg-best": {
                "parent": None,
                "prompts": {
                    "__module__": {
                        "type": "TEXT",
                        "text_template": "Base prompt",
                    },
                    "chat": {
                        "type": "LIST",
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "hi"},
                        ],
                    },
                },
            }
        },
    }

    report = OptimizationReport.from_runtime(runtime)
    inflated = inflate_prompts_from_report(report)

    assert "cfg-best" in inflated
    best_prompts = inflated["cfg-best"]

    # TEXT prompt
    text_prompt = best_prompts["__module__"]
    assert isinstance(text_prompt, Prompt)
    assert text_prompt.text_template == "Base prompt"

    # LIST prompt
    list_prompt = best_prompts["chat"]
    assert isinstance(list_prompt, Prompt)
    assert list_prompt.messages_template is not None
    assert len(list_prompt.messages_template) == 2

    roles = [m.role for m in list_prompt.messages_template]
    contents = [m.content for m in list_prompt.messages_template]
    assert roles == ["system", "user"]
    assert contents == ["sys", "hi"]


######################
# validate_instance  #
######################


def test_validate_instance_accepts_expected_type():
    value = "hello"

    result = validate_instance(
        component="MyComponent",
        param_name="param",
        value=value,
        expected_types=str,
    )

    # returns original value on success
    assert result is value


def test_validate_instance_accepts_tuple_of_expected_types():
    class A:
        pass

    class B:
        pass

    a = A()

    result = validate_instance(
        component="MyComponent",
        param_name="param",
        value=a,
        expected_types=(A, B),
    )

    assert result is a


def test_validate_instance_allows_none_when_flag_set():
    result = validate_instance(
        component="MyComponent",
        param_name="param",
        value=None,
        expected_types=str,
        allow_none=True,
    )

    assert result is None


def test_validate_instance_raises_for_wrong_type():
    with pytest.raises(DeepEvalError) as excinfo:
        validate_instance(
            component="MyComponent",
            param_name="param",
            value=123,
            expected_types=str,
        )

    msg = str(excinfo.value)
    assert "MyComponent expected `param` to be an instance of str" in msg
    assert "but received 'int' instead." in msg


########################
# validate_sequence_of #
########################


def test_validate_sequence_of_accepts_list_of_expected_type():
    items = [1, 2, 3]

    result = validate_sequence_of(
        component="MyComponent",
        param_name="items",
        value=items,
        expected_item_types=int,
    )

    # returns original container
    assert result is items


def test_validate_sequence_of_accepts_tuple_when_allowed():
    items = (1, 2, 3)

    result = validate_sequence_of(
        component="MyComponent",
        param_name="items",
        value=items,
        expected_item_types=int,
        sequence_types=(list, tuple),
    )

    assert result is items


def test_validate_sequence_of_allows_none_when_flag_set():
    result = validate_sequence_of(
        component="MyComponent",
        param_name="items",
        value=None,
        expected_item_types=int,
        allow_none=True,
    )

    assert result is None


def test_validate_sequence_of_rejects_none_without_allow():
    with pytest.raises(DeepEvalError) as excinfo:
        validate_sequence_of(
            component="MyComponent",
            param_name="goldens",
            value=None,
            expected_item_types=int,
        )

    msg = str(excinfo.value)
    # default sequence_types=(list, tuple)
    assert "MyComponent expected `goldens` to be a list or tuple of int" in msg
    assert "but received None instead." in msg


def test_validate_sequence_of_rejects_wrong_sequence_type():
    items = {1, 2, 3}  # set instead of list/tuple

    with pytest.raises(DeepEvalError) as excinfo:
        validate_sequence_of(
            component="MyComponent",
            param_name="items",
            value=items,
            expected_item_types=int,
        )

    msg = str(excinfo.value)
    assert "MyComponent expected `items` to be a list or tuple" in msg
    assert "but received 'set' instead." in msg


def test_validate_sequence_of_rejects_wrong_item_type():
    items = [1, "bad", 3]

    with pytest.raises(DeepEvalError) as excinfo:
        validate_sequence_of(
            component="MyComponent",
            param_name="items",
            value=items,
            expected_item_types=int,
        )

    msg = str(excinfo.value)
    assert (
        "MyComponent expected all elements of `items` to be instances of int"
        in msg
    )
    assert "element at index 1 has type 'str'." in msg
