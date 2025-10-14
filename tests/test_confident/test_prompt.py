import uuid
import time
from unittest.mock import patch
from deepeval.prompt import Prompt
from deepeval.prompt.api import (
    PromptType,
    PromptInterpolationType,
    PromptMessage,
    ModelSettings,
    ModelProvider,
    ReasoningEffort,
    OutputType,
    Verbosity,
)
from deepeval.confident.api import Api
from deepeval.metrics.faithfulness.schema import FaithfulnessVerdict


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
        assert prompt.text_template == f"Hello, world! {UUID}"
        assert prompt.messages_template is None
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.TEXT
        assert prompt.interpolation_type == PromptInterpolationType.FSTRING

    def test_push_with_interpolation_type(self):
        prompt = Prompt(alias=self.ALIAS_WITH_INTERPOLATION_TYPE)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Hello, world! {UUID}",
            interpolation_type=PromptInterpolationType.MUSTACHE,
        )

        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt.text_template == f"Hello, world! {UUID}"
        assert prompt.messages_template is None
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.TEXT
        assert prompt.interpolation_type == PromptInterpolationType.MUSTACHE

    def test_pull_by_label(self):
        """Test pulling text prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt.text_template is not None
        assert prompt.type == PromptType.TEXT
        assert prompt._prompt_version_id is not None
        assert prompt.interpolation_type is not None

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
        assert prompt_by_version.text_template is not None
        assert prompt_by_label.text_template is not None

    def test_cache_functionality(self):
        """Test that pulling from cache doesn't make API requests"""
        # First, cache a prompt by version
        prompt1 = Prompt(alias=self.ALIAS)
        prompt1.pull(write_to_cache=True)
        version = prompt1.version
        content = prompt1.text_template

        # Mock the API to verify no request is made
        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt2 = Prompt(alias=self.ALIAS)
            prompt2.pull(version=version, default_to_cache=True)

            # Verify content matches without API call
            assert prompt2.text_template == content
            assert prompt2.version == version
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

        # Test the same for label cache
        prompt3 = Prompt(alias=self.ALIAS)
        prompt3.pull(label=self.LABEL, write_to_cache=True)
        label_content = prompt3.text_template

        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt4 = Prompt(alias=self.ALIAS)
            prompt4.pull(label=self.LABEL, default_to_cache=True)

            # Verify content matches without API call
            assert prompt4.text_template == label_content
            assert prompt4.label == self.LABEL
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

    def test_version_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert spy_api.call_count == 3  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert spy_api.call_count == 3  # 1 for pull, 2 for polling
            prompt._stop_polling()


class TestPromptList:
    ALIAS = "test_prompt_list"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_list_interpolation_type"
    ALIAS_SETTINGS = "test_prompt_settings"
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
        assert prompt.text_template is None
        assert prompt.messages_template == messages
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.LIST
        assert prompt.interpolation_type == PromptInterpolationType.FSTRING

    def test_update(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Hello, assistant! {UUID}"),
            PromptMessage(role="assistant", content=f"Hello, user! {UUID}"),
        ]

        prompt.push(messages=messages)

        prompt.update(
            version="latest",
            messages=[
                PromptMessage(role="user", content="Hello, assistant!"),
                PromptMessage(role="assistant", content="Hello, user!"),
            ],
            model_settings=ModelSettings(
                provider=ModelProvider.OPEN_AI,
                name="o4-mini",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stop_sequence=["stop"],
                verbosity=Verbosity.LOW,
                reasoning_effort=ReasoningEffort.MINIMAL,
            ),
            interpolation_type=PromptInterpolationType.FSTRING,
            output_type=OutputType.SCHEMA,
            output_schema=FaithfulnessVerdict,
        )

        assert prompt.text_template is None
        assert len(prompt.messages_template) == 2
        assert prompt.messages_template[0].content == "Hello, assistant!"
        assert prompt.messages_template[1].content == "Hello, user!"
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.LIST
        assert prompt.interpolation_type == PromptInterpolationType.FSTRING

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
        assert prompt.text_template is None
        assert prompt.messages_template == messages
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.LIST
        assert prompt.interpolation_type == PromptInterpolationType.MUSTACHE

    def test_pull_by_label(self):
        """Test pulling list prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt.messages_template is not None
        assert prompt.type == PromptType.LIST
        assert prompt._prompt_version_id is not None
        assert prompt.interpolation_type is not None

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
        assert prompt_by_version.messages_template is not None
        assert prompt_by_label.messages_template is not None

    def test_cache_functionality(self):
        """Test that pulling from cache doesn't make API requests"""
        # First, cache a prompt by version
        prompt1 = Prompt(alias=self.ALIAS)
        prompt1.pull(write_to_cache=True)
        version = prompt1.version
        content = prompt1.messages_template

        # Mock the API to verify no request is made
        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt2 = Prompt(alias=self.ALIAS)
            prompt2.pull(version=version, default_to_cache=True)

            # Verify content matches without API call
            assert prompt2.messages_template == content
            assert prompt2.version == version
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

        # Test the same for label cache
        prompt3 = Prompt(alias=self.ALIAS)
        prompt3.pull(label=self.LABEL, write_to_cache=True)
        label_content = prompt3.messages_template

        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt4 = Prompt(alias=self.ALIAS)
            prompt4.pull(label=self.LABEL, default_to_cache=True)

            # Verify content matches without API call
            assert prompt4.messages_template == label_content
            assert prompt4.label == self.LABEL
            # Api() should not have been instantiated when using cache
            mock_api.assert_not_called()

    def test_version_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert spy_api.call_count == 3  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert spy_api.call_count == 3  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_model_settings_pull(self):
        prompt = Prompt(alias=self.ALIAS_SETTINGS)
        prompt.pull()
        assert prompt.model_settings is not None
        assert prompt.model_settings.provider == ModelProvider.OPEN_AI
        assert prompt.model_settings.name == "gpt-4o"
        assert prompt.model_settings.temperature == 0.5
        assert prompt.model_settings.max_tokens == 1000
        assert prompt.model_settings.top_p == 0.9
        assert prompt.model_settings.frequency_penalty == 0.1
        assert prompt.model_settings.presence_penalty == 0.1
        assert prompt.model_settings.stop_sequence == ["hi"]
        assert prompt.model_settings.reasoning_effort == ReasoningEffort.MINIMAL
        assert prompt.model_settings.verbosity == Verbosity.LOW
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        assert hasattr(prompt.output_schema, "__fields__") or hasattr(
            prompt.output_schema, "model_fields"
        )
        expected_fields = {"verdict", "reason"}
        actual_fields = set(prompt.output_schema.model_fields.keys())
        assert actual_fields == expected_fields

    def test_push_with_model_settings_and_output(self):
        prompt = Prompt(alias=self.ALIAS_SETTINGS)
        UUID = uuid.uuid4()
        messages = [
            PromptMessage(role="user", content=f"Test message {UUID}"),
            PromptMessage(role="assistant", content=f"Test response {UUID}"),
        ]
        model_settings = ModelSettings(
            provider=ModelProvider.OPEN_AI,
            name="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequence=["hi"],
            reasoning_effort=ReasoningEffort.MINIMAL,
            verbosity=Verbosity.LOW,
        )
        output_type = OutputType.SCHEMA
        output_schema = FaithfulnessVerdict
        prompt.push(
            messages=messages,
            model_settings=model_settings,
            output_type=output_type,
            output_schema=output_schema,
            _verbose=False,
        )
        assert prompt.model_settings is not None
        assert prompt.model_settings.provider == ModelProvider.OPEN_AI
        assert prompt.model_settings.name == "gpt-4o"
        assert prompt.model_settings.temperature == 0.5
        assert prompt.model_settings.max_tokens == 1000
        assert prompt.model_settings.top_p == 0.9
        assert prompt.model_settings.frequency_penalty == 0.1
        assert prompt.model_settings.presence_penalty == 0.1
        assert prompt.model_settings.stop_sequence == ["hi"]
        assert prompt.model_settings.reasoning_effort == ReasoningEffort.MINIMAL
        assert prompt.model_settings.verbosity == Verbosity.LOW
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        expected_fields = {"verdict", "reason"}
        actual_fields = set(prompt.output_schema.model_fields.keys())
        assert actual_fields == expected_fields

    def test_cache_preserves_settings(self):
        """Test that caching preserves model settings and output configuration"""
        # First, pull and cache a prompt
        prompt1 = Prompt(alias=self.ALIAS_SETTINGS)
        prompt1.pull(write_to_cache=True)

        original_model_settings = prompt1.model_settings
        original_output_type = prompt1.output_type
        original_output_schema = prompt1.output_schema

        # Load from cache
        prompt2 = Prompt(alias=self.ALIAS_SETTINGS)
        prompt2.pull(default_to_cache=True)

        # Verify settings are preserved
        if original_model_settings:
            assert prompt2.model_settings is not None
            assert (
                prompt2.model_settings.provider
                == original_model_settings.provider
            )
            assert prompt2.model_settings.name == original_model_settings.name
            assert (
                prompt2.model_settings.temperature
                == original_model_settings.temperature
            )

        assert prompt2.output_type == original_output_type
        if original_output_schema is not None:
            assert prompt2.output_schema is not None
            original_fields = set(original_output_schema.model_fields.keys())
            cached_fields = set(prompt2.output_schema.model_fields.keys())
            assert cached_fields == original_fields
        else:
            assert prompt2.output_schema == original_output_schema
