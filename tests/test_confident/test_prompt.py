import pytest
import uuid
from unittest.mock import patch
from deepeval.prompt import Prompt
from deepeval.prompt.api import (
    PromptType,
    PromptInterpolationType,
    PromptMessage,
)


class TestPromptText:
    ALIAS = "test_prompt_text"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_text_interpolation_type"
    LABEL = "STAGING"
    LABEL_VERSION = "00.06.95"

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

    def test_pull_by_label(self):
        """Test pulling text prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt._text_template is not None
        assert prompt._type == PromptType.TEXT
        assert prompt._prompt_version_id is not None
        assert prompt._interpolation_type is not None

    def test_version_vs_label_pull(self):
        """Test that version and label pulls work independently"""
        # Pull by version (latest)
        prompt_by_version = Prompt(alias=self.ALIAS)
        prompt_by_version.pull()

        # Pull by label
        prompt_by_label = Prompt(alias=self.ALIAS)
        prompt_by_label.pull(label=self.LABEL)

        # Version pull should not have label
        assert prompt_by_version.label is None
        assert prompt_by_version.version is not None

        # Label pull should have both
        assert prompt_by_label.label == self.LABEL
        assert prompt_by_label.version == self.LABEL_VERSION

        # Both should have valid content
        assert prompt_by_version._text_template is not None
        assert prompt_by_label._text_template is not None

    def test_cache_functionality(self):
        """Test that pulling from cache doesn't make API requests"""
        # First, cache a prompt by version
        prompt1 = Prompt(alias=self.ALIAS)
        prompt1.pull(write_to_cache=True)
        version = prompt1.version
        content = prompt1._text_template

        # Mock the API to verify no request is made
        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt2 = Prompt(alias=self.ALIAS)
            prompt2.pull(version=version, default_to_cache=True)

            # Verify content matches without API call
            assert prompt2._text_template == content
            assert prompt2.version == version
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

        # Test the same for label cache
        prompt3 = Prompt(alias=self.ALIAS)
        prompt3.pull(label=self.LABEL, write_to_cache=True)
        label_content = prompt3._text_template

        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt4 = Prompt(alias=self.ALIAS)
            prompt4.pull(label=self.LABEL, default_to_cache=True)

            # Verify content matches without API call
            assert prompt4._text_template == label_content
            assert prompt4.label == self.LABEL
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()


class TestPromptList:
    ALIAS = "test_prompt_list"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_list_interpolation_type"
    LABEL = "STAGING"
    LABEL_VERSION = "00.07.01"

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

    def test_pull_by_label(self):
        """Test pulling list prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt._messages_template is not None
        assert prompt._type == PromptType.LIST
        assert prompt._prompt_version_id is not None
        assert prompt._interpolation_type is not None

    def test_version_vs_label_pull(self):
        """Test that version and label pulls work independently"""
        # Pull by version (latest)
        prompt_by_version = Prompt(alias=self.ALIAS)
        prompt_by_version.pull()

        # Pull by label
        prompt_by_label = Prompt(alias=self.ALIAS)
        prompt_by_label.pull(label=self.LABEL)

        # Version pull should not have label
        assert prompt_by_version.label is None
        assert prompt_by_version.version is not None

        # Label pull should have both
        assert prompt_by_label.label == self.LABEL
        assert prompt_by_label.version == self.LABEL_VERSION

        # Both should have valid content
        assert prompt_by_version._messages_template is not None
        assert prompt_by_label._messages_template is not None

    def test_cache_functionality(self):
        """Test that pulling from cache doesn't make API requests"""
        # First, cache a prompt by version
        prompt1 = Prompt(alias=self.ALIAS)
        prompt1.pull(write_to_cache=True)
        version = prompt1.version
        content = prompt1._messages_template

        # Mock the API to verify no request is made
        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt2 = Prompt(alias=self.ALIAS)
            prompt2.pull(version=version, default_to_cache=True)

            # Verify content matches without API call
            assert prompt2._messages_template == content
            assert prompt2.version == version
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

        # Test the same for label cache
        prompt3 = Prompt(alias=self.ALIAS)
        prompt3.pull(label=self.LABEL, write_to_cache=True)
        label_content = prompt3._messages_template

        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt4 = Prompt(alias=self.ALIAS)
            prompt4.pull(label=self.LABEL, default_to_cache=True)

            # Verify content matches without API call
            assert prompt4._messages_template == label_content
            assert prompt4.label == self.LABEL
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()
