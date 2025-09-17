import pytest
import uuid
from deepeval.prompt import Prompt
from deepeval.prompt.api import (
    PromptType,
    PromptInterpolationType,
    PromptMessage,
)


class TestPromptText:
    ALIAS = "test_prompt_text"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_text_interpolation_type"

    def test_push(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID = uuid.uuid4()

        # generate uuid
        prompt.push(text=f"Hello, world! {UUID}")

        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt._text_template == f"Hello, world! {UUID}"
        assert prompt._messages_template is None
        assert prompt._prompt_version_id is not None
        assert prompt._type == PromptType.TEXT
        assert prompt._interpolation_type == PromptInterpolationType.FSTRING

    def test_push_with_interpolation_type(self):
        prompt = Prompt(alias=self.ALIAS_WITH_INTERPOLATION_TYPE)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Hello, world! {UUID}",
            interpolation_type=PromptInterpolationType.MUSTACHE,
        )

        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt._text_template == f"Hello, world! {UUID}"
        assert prompt._messages_template is None
        assert prompt._prompt_version_id is not None
        assert prompt._type == PromptType.TEXT
        assert prompt._interpolation_type == PromptInterpolationType.MUSTACHE


class TestPromptList:
    ALIAS = "test_prompt_list"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_list_interpolation_type"

    def test_push(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Hello, assistant! {UUID}"),
            PromptMessage(role="assistant", content=f"Hello, user! {UUID}"),
        ]

        prompt.push(messages=messages)
        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt._text_template is None
        assert prompt._messages_template == messages
        assert prompt._prompt_version_id is not None
        assert prompt._type == PromptType.LIST
        assert prompt._interpolation_type == PromptInterpolationType.FSTRING

    def test_push_with_interpolation_type(self):
        prompt = Prompt(alias=self.ALIAS_WITH_INTERPOLATION_TYPE)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Hello, assistant! {UUID}"),
            PromptMessage(role="assistant", content=f"Hello, user! {UUID}"),
        ]

        prompt.push(
            messages=messages,
            interpolation_type=PromptInterpolationType.MUSTACHE,
        )

        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt._text_template is None
        assert prompt._messages_template == messages
        assert prompt._prompt_version_id is not None
        assert prompt._type == PromptType.LIST
        assert prompt._interpolation_type == PromptInterpolationType.MUSTACHE
