import pytest
from deepeval.errors import DeepEvalError
from deepeval.optimizer.utils import _parse_prompt, _create_prompt
from deepeval.prompt.prompt import Prompt
from deepeval.prompt import PromptMessage


def test_parse_prompt_text_returns_template():
    prompt = Prompt(text_template="Hello {input}")
    assert _parse_prompt(prompt) == "Hello {input}"


def test_parse_prompt_list_returns_json_string():
    prompt = Prompt(
        messages_template=[
            PromptMessage(role="system", content="You are helpful."),
            PromptMessage(role="user", content="Q: {input}"),
        ]
    )
    out = _parse_prompt(prompt)
    assert '"role": "system"' in out
    assert '"content": "Q: {input}"' in out


def test_create_prompt_list_accepts_json_array():
    old_prompt = Prompt(
        messages_template=[
            PromptMessage(role="system", content="old"),
            PromptMessage(role="user", content="{input}"),
        ]
    )
    new_content = (
        '[{"role":"system","content":"new system"},'
        '{"role":"user","content":"new user"}]'
    )

    new_prompt = _create_prompt(old_prompt, new_content)
    assert new_prompt.messages_template is not None
    assert len(new_prompt.messages_template) == 2
    assert new_prompt.messages_template[0].content == "new system"


def test_create_prompt_list_rejects_comma_separated_objects_without_array():
    old_prompt = Prompt(
        messages_template=[
            PromptMessage(role="system", content="old"),
            PromptMessage(role="user", content="{input}"),
        ]
    )
    new_content = (
        '{"role":"system","content":"new system"},'
        '{"role":"user","content":"new user"}'
    )

    with pytest.raises(
        DeepEvalError, match="Failed to parse the LLM's rewritten messages into JSON"
    ):
        _create_prompt(old_prompt, new_content)


def test_create_prompt_list_raises_for_invalid_json():
    old_prompt = Prompt(
        messages_template=[
            PromptMessage(role="system", content="old"),
            PromptMessage(role="user", content="{input}"),
        ]
    )
    with pytest.raises(
        DeepEvalError, match="Failed to parse the LLM's rewritten messages into JSON"
    ):
        _create_prompt(old_prompt, "not-json-at-all")
