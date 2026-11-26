import json

from deepeval.prompt.api import PromptMessage, PromptType
from deepeval.prompt.prompt import Prompt


def test_load_text_then_interpolate(tmp_path):
    file_path = tmp_path / "greeting.txt"
    file_path.write_text("Hello, {name}!", encoding="utf-8")
    prompt = Prompt()

    prompt.load(str(file_path))

    assert prompt.type == PromptType.TEXT
    assert prompt.messages_template is None
    assert prompt.interpolate(name="Ada") == "Hello, Ada!"


def test_load_json_messages_then_interpolate(tmp_path):
    file_path = tmp_path / "greeting.json"
    messages = [{"role": "user", "content": "Hello, {name}!"}]
    file_path.write_text(json.dumps(messages), encoding="utf-8")
    prompt = Prompt()

    prompt.load(str(file_path))

    assert prompt.type == PromptType.LIST
    assert prompt.text_template is None
    assert prompt.interpolate(name="Ada") == [
        {"role": "user", "content": "Hello, Ada!"}
    ]


def test_load_json_dict_with_messages_key_then_interpolate(tmp_path):
    file_path = tmp_path / "greeting.json"
    data = {"messages": [{"role": "user", "content": "Hello, {name}!"}]}
    file_path.write_text(json.dumps(data), encoding="utf-8")
    prompt = Prompt(text_template="Previous text")

    prompt.load(str(file_path), messages_key="messages")

    assert prompt.type == PromptType.LIST
    assert prompt.text_template is None
    assert prompt.interpolate(name="Ada") == [
        {"role": "user", "content": "Hello, Ada!"}
    ]


def test_load_text_replaces_messages(tmp_path):
    file_path = tmp_path / "greeting.txt"
    file_path.write_text("Hello, {name}!", encoding="utf-8")
    prompt = Prompt(
        messages_template=[
            PromptMessage(role="user", content="Previous message")
        ]
    )
    assert prompt.type == PromptType.LIST

    prompt.load(str(file_path))

    assert prompt.type == PromptType.TEXT
    assert prompt.messages_template is None
    assert prompt.interpolate(name="Ada") == "Hello, Ada!"
