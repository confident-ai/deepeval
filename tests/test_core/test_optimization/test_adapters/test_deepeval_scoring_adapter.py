from __future__ import annotations

from typing import Dict, List

import pytest

from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.errors import DeepEvalError
from deepeval.optimization.adapters.deepeval_scoring_adapter import (
    DeepEvalScoringAdapter,
)
from deepeval.prompt.api import PromptType, PromptMessage
from deepeval.prompt.prompt import Prompt
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
)


###############################
# _primary_input_from_golden()
###############################


def test_primary_input_from_golden_uses_input_for_golden():
    adapter = DeepEvalScoringAdapter()
    golden = Golden(input="plain input", scenario="should_be_ignored")

    primary = adapter._primary_input_from_golden(golden)

    assert primary == "plain input"


def test_primary_input_from_golden_uses_scenario_for_conversational_golden():
    adapter = DeepEvalScoringAdapter()
    conv = ConversationalGolden(
        scenario="conversational scenario",
        input="should_be_ignored",
    )

    primary = adapter._primary_input_from_golden(conv)

    assert primary == "conversational scenario"


def test_primary_input_from_golden_rejects_unknown_type():
    adapter = DeepEvalScoringAdapter()
    with pytest.raises(DeepEvalError):
        adapter._primary_input_from_golden(object())


#########################
# _compile_prompt_text()
#########################


def test_compile_prompt_text_appends_input_when_present():
    adapter = DeepEvalScoringAdapter()

    prompt = Prompt(text_template="Base prompt")
    # Be explicit about type to avoid depending on Prompt internals.
    prompt.type = PromptType.TEXT

    golden = Golden(input="question about cats")

    compiled = adapter._compile_prompt_text(prompt, golden)

    assert compiled == "Base prompt\n\nquestion about cats"


def test_compile_prompt_text_returns_base_when_input_empty():
    adapter = DeepEvalScoringAdapter()

    prompt = Prompt(text_template="Base prompt")
    prompt.type = PromptType.TEXT

    golden = Golden(input="")

    compiled = adapter._compile_prompt_text(prompt, golden)

    assert compiled == "Base prompt"


#############################
# _compile_prompt_messages()
#############################


def test_compile_prompt_messages_appends_input_with_configured_role():
    adapter = DeepEvalScoringAdapter(list_input_role="end_user")

    prompt = Prompt(
        messages_template=[
            PromptMessage(role="system", content="sys"),
            PromptMessage(role="assistant", content="assistant content"),
        ]
    )
    prompt.type = PromptType.LIST

    golden = Golden(input="user question")

    messages = adapter._compile_prompt_messages(prompt, golden)

    assert len(messages) == 3
    # Original messages preserved
    assert messages[0].role == "system"
    assert messages[0].content == "sys"
    assert messages[1].role == "assistant"
    assert messages[1].content == "assistant content"
    # New message appended with configured role and golden input
    assert messages[2].role == "end_user"
    assert messages[2].content == "user question"


##############################
# _default_build_test_case()
##############################


def test_default_build_test_case_for_golden_maps_core_fields():
    adapter = DeepEvalScoringAdapter()

    golden = Golden(
        input="Q?",
        expected_output="A",
        context=["ctx1"],
        retrieval_context=["doc1"],
        additional_metadata={"tag": "value"},
        comments="note",
        name="case-1",
        tools_called=[ToolCall(name="tool1")],
        expected_tools=[ToolCall(name="tool2")],
    )

    actual = "model answer"

    test_case = adapter._default_build_test_case(golden, actual)

    assert isinstance(test_case, LLMTestCase)
    assert test_case.input == golden.input
    assert test_case.expected_output == golden.expected_output
    assert test_case.actual_output == actual
    assert test_case.context == golden.context
    assert test_case.retrieval_context == golden.retrieval_context
    assert test_case.additional_metadata == golden.additional_metadata
    assert test_case.comments == golden.comments
    assert test_case.name == golden.name
    assert test_case.tools_called == golden.tools_called
    assert test_case.expected_tools == golden.expected_tools


def test_default_build_test_case_conversational_no_turns_synthesizes_two_turns():
    adapter = DeepEvalScoringAdapter()

    conv = ConversationalGolden(
        scenario="user scenario",
        expected_outcome="expected",
        user_description="description",
        context=["ctx"],
        additional_metadata={"key": "value"},
        comments="note",
        name="conv-case",
        turns=None,
    )
    actual = "assistant reply"

    test_case = adapter._default_build_test_case(conv, actual)

    assert isinstance(test_case, ConversationalTestCase)
    assert test_case.scenario == conv.scenario
    assert test_case.expected_outcome == conv.expected_outcome
    assert test_case.user_description == conv.user_description
    assert test_case.context == conv.context
    assert test_case.additional_metadata == conv.additional_metadata
    assert test_case.comments == conv.comments
    assert test_case.name == conv.name

    turns = test_case.turns
    assert len(turns) == 2

    # Synthesized conversation: [user, assistant]
    assert turns[0].role == "user"
    assert turns[0].content == conv.scenario
    assert turns[1].role == "assistant"
    assert turns[1].content == actual


def test_default_build_test_case_conversational_replaces_last_assistant_turn():
    adapter = DeepEvalScoringAdapter()

    turns: List[Turn] = [
        Turn(role="user", content="hi"),
        Turn(role="assistant", content="old answer"),
    ]
    conv = ConversationalGolden(
        scenario="ignored here",
        expected_outcome="expected",
        user_description=None,
        context=None,
        additional_metadata=None,
        comments=None,
        name="conv-with-turns",
        turns=turns,
    )
    actual = "fresh answer"

    test_case = adapter._default_build_test_case(conv, actual)

    assert isinstance(test_case, ConversationalTestCase)
    assert len(test_case.turns) == 2
    assert test_case.turns[0].role == "user"
    assert test_case.turns[0].content == "hi"

    # Last assistant turn is replaced, not appended
    last = test_case.turns[1]
    assert last.role == "assistant"
    assert last.content == "fresh answer"


def test_default_build_test_case_conversational_appends_assistant_when_last_not_assistant():
    adapter = DeepEvalScoringAdapter()

    turns: List[Turn] = [Turn(role="user", content="only user so far")]
    conv = ConversationalGolden(
        scenario="ignored here",
        expected_outcome=None,
        user_description=None,
        context=None,
        additional_metadata=None,
        comments=None,
        name="conv-append-assistant",
        turns=turns,
    )
    actual = "assistant now"

    test_case = adapter._default_build_test_case(conv, actual)

    assert isinstance(test_case, ConversationalTestCase)
    assert len(test_case.turns) == 2

    assert test_case.turns[0].role == "user"
    assert test_case.turns[0].content == "only user so far"

    last = test_case.turns[1]
    assert last.role == "assistant"
    assert last.content == "assistant now"


##################################
# _select_module_id_from_prompts #
##################################


def _make_text_prompt(text: str) -> Prompt:
    p = Prompt(text_template=text)
    p.type = PromptType.TEXT
    return p


def test_select_module_prefers_default_module_id():
    adapter = DeepEvalScoringAdapter()

    p_default = _make_text_prompt("default")
    p_other = _make_text_prompt("other")

    prompts_by_module: Dict[str, Prompt] = {
        adapter.DEFAULT_MODULE_ID: p_default,
        "other_module": p_other,
    }

    module_id = adapter._select_module_id_from_prompts(prompts_by_module)

    assert module_id == adapter.DEFAULT_MODULE_ID


def test_select_module_falls_back_to_first_key_when_default_missing():
    adapter = DeepEvalScoringAdapter()

    p_a = _make_text_prompt("A")
    p_b = _make_text_prompt("B")

    prompts_by_module: Dict[str, Prompt] = {
        "module_a": p_a,
        "module_b": p_b,
    }

    module_id = adapter._select_module_id_from_prompts(prompts_by_module)

    # Python dicts preserve insertion order; we expect the first key.
    assert module_id == "module_a"


def test_select_module_raises_for_empty_prompts_dict():
    adapter = DeepEvalScoringAdapter()

    with pytest.raises(DeepEvalError):
        adapter._select_module_id_from_prompts({})
