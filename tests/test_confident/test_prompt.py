import uuid
import time
from pydantic import BaseModel
from unittest.mock import patch
from deepeval.prompt import Prompt, Tool
from deepeval.prompt.api import (
    PromptType,
    PromptInterpolationType,
    PromptMessage,
    ModelSettings,
    ModelProvider,
    ReasoningEffort,
    OutputType,
    Verbosity,
    ToolMode,
)
from deepeval.confident.api import Api
from deepeval.metrics.faithfulness.schema import FaithfulnessVerdict


class NestedObject(BaseModel):
    nested_field: str
    nested_number: int


class SimpleSchema(BaseModel):
    name: str
    value: float


class ComplexOutputSchema(BaseModel):
    title: str
    count: int
    score: float
    active: bool
    metadata: NestedObject


class DeeplyNestedObject(BaseModel):
    level3_field: str


class MiddleNestedObject(BaseModel):
    level2_field: int
    deep_object: DeeplyNestedObject


class VeryComplexSchema(BaseModel):
    id: str
    simple_field: str
    number_field: int
    float_field: float
    bool_field: bool
    nested_obj: MiddleNestedObject


class ToolInputSchema(BaseModel):
    query: str
    max_results: int
    include_metadata: bool


class UpdatedToolInputSchema(BaseModel):
    query: str
    max_results: int
    include_metadata: bool
    new_field: str


class TestPromptText:
    ALIAS = "test_prompt_text"
    ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_text_interpolation_type"
    LABEL = "STAGING"
    LABEL_VERSION = "00.06.95"

    def test_push(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID = str(uuid.uuid4())

        TEXT = f"Hello, world! {UUID}"

        # generate uuid
        prompt.push(text=TEXT)

        prompt.pull()

        assert prompt.version[0] == "0"
        assert prompt.text_template == TEXT
        assert prompt.messages_template is None
        assert prompt._prompt_version_id is not None
        assert prompt.type == PromptType.TEXT
        assert prompt.interpolation_type == PromptInterpolationType.FSTRING

    def test_push_with_interpolation_type(self):
        prompt = Prompt(alias=self.ALIAS_WITH_INTERPOLATION_TYPE)

        UUID = str(uuid.uuid4())
        TEXT = f"Hello, world! {UUID}"

        prompt.push(
            text=TEXT,
            interpolation_type=PromptInterpolationType.MUSTACHE,
        )

        prompt.pull(default_to_cache=False)

        assert prompt.version[0] == "0"
        assert prompt.text_template == TEXT
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

    def test_get_versions(self):
        """Test get versions for text prompt"""
        prompt = Prompt(alias=self.ALIAS)

        versions = prompt._get_versions()
        assert versions is not None

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

            assert spy_api.call_count >= 2  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert spy_api.call_count >= 2  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_push_with_simple_output_schema(self):
        """Test pushing text prompt with simple output schema"""
        ALIAS = "test_prompt_text_simple_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Generate data {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=SimpleSchema,
        )

        prompt.pull()

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        assert hasattr(prompt.output_schema, "model_fields")

        expected_fields = {"name", "value"}
        actual_fields = set(prompt.output_schema.model_fields.keys())
        assert actual_fields == expected_fields

        # Verify field types
        assert prompt.output_schema.model_fields["name"].annotation == str
        assert prompt.output_schema.model_fields["value"].annotation == float

    def test_push_with_nested_output_schema(self):
        """Test pushing text prompt with nested output schema"""
        ALIAS = "test_prompt_text_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Generate complex data {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
        )

        prompt.pull()

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None

        expected_fields = {"title", "count", "score", "active", "metadata"}
        actual_fields = set(prompt.output_schema.model_fields.keys())
        assert actual_fields == expected_fields

        # Verify nested object
        nested_type = prompt.output_schema.model_fields["metadata"].annotation
        assert hasattr(nested_type, "model_fields")
        nested_fields = set(nested_type.model_fields.keys())
        assert nested_fields == {"nested_field", "nested_number"}

    def test_push_with_deeply_nested_output_schema(self):
        """Test pushing text prompt with deeply nested output schema (3 levels)"""
        ALIAS = "test_prompt_text_deep_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Generate very complex data {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=VeryComplexSchema,
        )

        prompt.pull()

        # Verify top level schema
        assert prompt.output_schema is not None
        top_fields = set(prompt.output_schema.model_fields.keys())
        assert top_fields == {
            "id",
            "simple_field",
            "number_field",
            "float_field",
            "bool_field",
            "nested_obj",
        }

        # Verify level 2 nested object
        level2_type = prompt.output_schema.model_fields["nested_obj"].annotation
        assert hasattr(level2_type, "model_fields")
        level2_fields = set(level2_type.model_fields.keys())
        assert level2_fields == {"level2_field", "deep_object"}

        # Verify level 3 nested object
        level3_type = level2_type.model_fields["deep_object"].annotation
        assert hasattr(level3_type, "model_fields")
        level3_fields = set(level3_type.model_fields.keys())
        assert level3_fields == {"level3_field"}

    def test_push_single_tool(self):
        """Test pushing text prompt with a single tool"""
        ALIAS = "test_prompt_text_single_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        tool = Tool(
            name="SearchTool",
            description="A tool for searching",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            text=f"Use the search tool {UUID}",
            tools=[tool],
        )
        prompt.tools = None
        prompt.pull(default_to_cache=False)

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        pulled_tool = prompt.tools[0]
        assert pulled_tool.name == "SearchTool"
        assert pulled_tool.description == "A tool for searching"
        assert pulled_tool.mode == ToolMode.STRICT

        # Verify tool schema
        assert pulled_tool.structured_schema is not None
        assert pulled_tool.structured_schema.fields is not None

        # Check input_schema property
        input_schema = pulled_tool.input_schema
        assert input_schema["type"] == "object"
        assert "query" in input_schema["properties"]
        assert "max_results" in input_schema["properties"]
        assert "include_metadata" in input_schema["properties"]

    def test_push_multiple_tools(self):
        """Test pushing text prompt with multiple tools"""
        ALIAS = "test_prompt_text_multiple_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        tool1 = Tool(
            name="SearchTool",
            description="Search tool",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        tool2 = Tool(
            name="AnalysisTool",
            description="Analysis tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            text=f"Use multiple tools {UUID}",
            tools=[tool1, tool2],
        )

        prompt.pull()

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 2

        tool_names = {tool.name for tool in prompt.tools}
        assert tool_names == {"SearchTool", "AnalysisTool"}

        # Verify each tool
        for tool in prompt.tools:
            assert tool.structured_schema is not None
            assert tool.input_schema is not None

    def test_update_tool_by_name(self):
        """Test updating a tool with the same name (should replace it)"""
        ALIAS = "test_prompt_text_update_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        # Push initial tool
        original_tool = Tool(
            name="SearchTool",
            description="Original search tool",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            text=f"Initial tool push {UUID}",
            tools=[original_tool],
        )

        prompt.pull()

        initial_tool = prompt.tools[0]
        assert initial_tool.description == "Original search tool"

        # Update with new tool (same name)
        updated_tool = Tool(
            name="SearchTool",  # Same name
            description="Updated search tool",  # Different description
            mode=ToolMode.NO_ADDITIONAL,  # Different mode
            structured_schema=UpdatedToolInputSchema,  # Different schema
        )

        prompt.update(
            version="latest",
            tools=[updated_tool],
        )
        prompt.tools = None
        prompt.pull(default_to_cache=False)

        # Verify tool was updated
        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        final_tool = prompt.tools[0]
        assert final_tool.name == "SearchTool"
        assert final_tool.description == "Updated search tool"
        assert final_tool.mode == ToolMode.NO_ADDITIONAL

        # Verify schema was updated
        input_schema = final_tool.input_schema
        assert "new_field" in input_schema["properties"]

    def test_push_output_schema_and_tools(self):
        """Test pushing both output schema and tools together"""
        ALIAS = "test_prompt_text_schema_and_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        tool = Tool(
            name="DataTool",
            description="Data processing tool",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            text=f"Process data with tool {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
            tools=[tool],
        )
        prompt.output_schema = None
        prompt.tools = None
        prompt.pull()

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        assert "title" in prompt.output_schema.model_fields

        # Verify tool
        assert prompt.tools is not None
        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == "DataTool"

    def test_pull_preserves_tool_details(self):
        """Test that pulling preserves all tool details including schema structure"""
        ALIAS = "test_prompt_text_tool_preservation"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        tool = Tool(
            name="DetailedTool",
            description="A tool with detailed schema",
            mode=ToolMode.STRICT,
            structured_schema=VeryComplexSchema,
        )

        prompt.push(
            text=f"Detailed tool test {UUID}",
            tools=[tool],
        )

        # Pull multiple times to ensure consistency
        for _ in range(3):
            prompt.pull()

            assert prompt.tools is not None
            assert len(prompt.tools) == 1

            pulled_tool = prompt.tools[0]
            assert pulled_tool.name == "DetailedTool"
            assert pulled_tool.description == "A tool with detailed schema"
            assert pulled_tool.mode == ToolMode.STRICT

            # Verify input schema has all fields
            input_schema = pulled_tool.input_schema
            assert "id" in input_schema["properties"]
            assert "simple_field" in input_schema["properties"]
            assert "nested_obj" in input_schema["properties"]

            # Verify nested structure
            nested_props = input_schema["properties"]["nested_obj"][
                "properties"
            ]
            assert "level2_field" in nested_props
            assert "deep_object" in nested_props

    def test_cache_preserves_output_schema_and_tools(self):
        """Test that caching preserves output schema and tools"""
        ALIAS = "test_prompt_text_cache_schema_tools"
        prompt1 = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        tool = Tool(
            name="CachedTool",
            description="Tool for cache test",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt1.push(
            text=f"Cache test {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
            tools=[tool],
        )

        # Pull and cache
        prompt1.pull(write_to_cache=False)
        version = prompt1.version

        # Load from cache
        prompt2 = Prompt(alias=ALIAS)
        prompt2.pull(version=version)

        # Verify output schema preserved
        assert prompt2.output_schema is not None
        assert set(prompt2.output_schema.model_fields.keys()) == set(
            prompt1.output_schema.model_fields.keys()
        )

        # Verify tools preserved
        assert prompt2.tools is not None
        assert len(prompt2.tools) == len(prompt1.tools)
        assert prompt2.tools[0].name == prompt1.tools[0].name
        assert prompt2.tools[0].mode == prompt1.tools[0].mode


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

        prompt.pull(default_to_cache=False)

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

    def test_get_versions(self):
        """Test get versions for list prompt"""
        prompt = Prompt(alias=self.ALIAS)

        versions = prompt._get_versions()
        assert versions is not None

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

            assert spy_api.call_count >= 2  # 1 for pull, 2 for polling
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2, default_to_cache=False)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert spy_api.call_count >= 2  # 1 for pull, 2 for polling
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
        assert hasattr(prompt.output_schema, "model_fields")
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

    def test_push_with_simple_output_schema(self):
        """Test pushing list prompt with simple output schema"""
        ALIAS = "test_prompt_list_simple_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Generate data {UUID}"),
            PromptMessage(role="assistant", content=f"Here's the data {UUID}"),
        ]

        prompt.push(
            messages=messages,
            output_type=OutputType.SCHEMA,
            output_schema=SimpleSchema,
        )

        prompt.pull()

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None

        expected_fields = {"name", "value"}
        actual_fields = set(prompt.output_schema.model_fields.keys())
        assert actual_fields == expected_fields

    def test_push_with_nested_output_schema(self):
        """Test pushing list prompt with nested output schema"""
        ALIAS = "test_prompt_list_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="system", content="You are a data generator"),
            PromptMessage(role="user", content=f"Generate complex data {UUID}"),
        ]

        prompt.push(
            messages=messages,
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
        )

        prompt.pull()

        # Verify nested structure
        assert prompt.output_schema is not None
        assert "metadata" in prompt.output_schema.model_fields

        nested_type = prompt.output_schema.model_fields["metadata"].annotation
        assert hasattr(nested_type, "model_fields")
        assert "nested_field" in nested_type.model_fields
        assert "nested_number" in nested_type.model_fields

    def test_push_with_deeply_nested_output_schema(self):
        """Test pushing list prompt with deeply nested output schema"""
        ALIAS = "test_prompt_list_deep_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Complex nested data {UUID}"),
        ]

        prompt.push(
            messages=messages,
            output_type=OutputType.SCHEMA,
            output_schema=VeryComplexSchema,
        )

        prompt.pull()

        # Verify 3-level nesting
        assert prompt.output_schema is not None

        # Level 1
        assert "nested_obj" in prompt.output_schema.model_fields

        # Level 2
        level2_type = prompt.output_schema.model_fields["nested_obj"].annotation
        assert "deep_object" in level2_type.model_fields

        # Level 3
        level3_type = level2_type.model_fields["deep_object"].annotation
        assert "level3_field" in level3_type.model_fields

    def test_push_single_tool(self):
        """Test pushing list prompt with a single tool"""
        ALIAS = "test_prompt_list_single_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Use search tool {UUID}"),
        ]

        tool = Tool(
            name="SearchTool",
            description="Search functionality",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            messages=messages,
            tools=[tool],
        )
        prompt.tools = None
        prompt.pull(default_to_cache=False)

        # Verify tool
        assert prompt.tools is not None
        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == "SearchTool"

        # Verify tool schema
        input_schema = prompt.tools[0].input_schema
        assert "query" in input_schema["properties"]
        assert input_schema["properties"]["query"]["type"] == "string"

    def test_push_multiple_tools(self):
        """Test pushing list prompt with multiple tools"""
        ALIAS = "test_prompt_list_multiple_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Multiple tools test {UUID}"),
        ]

        tool1 = Tool(
            name="Tool1",
            description="First tool",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        tool2 = Tool(
            name="Tool2",
            description="Second tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=ToolInputSchema,
        )

        tool3 = Tool(
            name="Tool3",
            description="Third tool",
            mode=ToolMode.ALLOW_ADDITIONAL,
            structured_schema=ComplexOutputSchema,
        )

        prompt.push(
            messages=messages,
            tools=[tool1, tool2, tool3],
        )

        prompt.pull()

        # Verify all tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 3

        tool_names = {tool.name for tool in prompt.tools}
        assert tool_names == {"Tool1", "Tool2", "Tool3"}

        # Verify different modes
        modes = {tool.name: tool.mode for tool in prompt.tools}
        assert modes["Tool1"] == ToolMode.STRICT
        assert modes["Tool2"] == ToolMode.NO_ADDITIONAL
        assert modes["Tool3"] == ToolMode.ALLOW_ADDITIONAL

    def test_update_tool_by_name(self):
        """Test updating a tool in list prompt"""
        ALIAS = "test_prompt_list_update_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Initial {UUID}"),
        ]

        # Initial tool
        tool = Tool(
            name="UpdateableTool",
            description="Original",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(messages=messages, tools=[tool])
        prompt.pull()

        assert prompt.tools[0].description == "Original"

        # Update tool
        updated_tool = Tool(
            name="UpdateableTool",
            description="Updated",
            mode=ToolMode.ALLOW_ADDITIONAL,
            structured_schema=UpdatedToolInputSchema,
        )

        prompt.update(
            version="latest",
            tools=[updated_tool],
        )

        prompt.pull()

        assert prompt.tools[0].description == "Updated"
        assert prompt.tools[0].mode == ToolMode.ALLOW_ADDITIONAL

        # Verify new field in schema
        input_schema = prompt.tools[0].input_schema
        assert "new_field" in input_schema["properties"]

    def test_push_output_schema_and_tools(self):
        """Test pushing list prompt with both output schema and tools"""
        ALIAS = "test_prompt_list_schema_and_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="system", content="You are helpful"),
            PromptMessage(role="user", content=f"Process {UUID}"),
        ]

        tool = Tool(
            name="ProcessorTool",
            description="Processing tool",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            messages=messages,
            output_type=OutputType.SCHEMA,
            output_schema=VeryComplexSchema,
            tools=[tool],
        )
        prompt.output_schema = None
        prompt.tools = None
        prompt.pull()

        # Verify both present
        assert prompt.output_schema is not None
        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        # Verify they're different schemas
        output_fields = set(prompt.output_schema.model_fields.keys())
        tool_input_schema = prompt.tools[0].input_schema
        tool_fields = set(tool_input_schema["properties"].keys())

        # They should have different fields
        assert output_fields != tool_fields

    def test_pull_preserves_tool_details(self):
        """Test that pulling list prompt preserves all tool details"""
        ALIAS = "test_prompt_list_tool_preservation"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Preserve test {UUID}"),
        ]

        tool = Tool(
            name="ComplexTool",
            description="Tool with complex schema",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=VeryComplexSchema,
        )

        prompt.push(messages=messages, tools=[tool])

        # Pull multiple times
        for i in range(3):
            prompt.pull()

            assert prompt.tools is not None
            pulled_tool = prompt.tools[0]

            assert pulled_tool.name == "ComplexTool"
            assert pulled_tool.mode == ToolMode.NO_ADDITIONAL

            # Verify complex nested structure preserved
            input_schema = pulled_tool.input_schema
            assert "nested_obj" in input_schema["properties"]

            nested = input_schema["properties"]["nested_obj"]
            assert nested["type"] == "object"
            assert "deep_object" in nested["properties"]

    def test_cache_preserves_output_schema_and_tools(self):
        """Test that caching preserves output schema and tools for list prompts"""
        ALIAS = "test_prompt_list_cache_schema_tools"
        prompt1 = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Cache test {UUID}"),
        ]

        tool = Tool(
            name="CachedListTool",
            description="Tool for list cache test",
            mode=ToolMode.STRICT,
            structured_schema=ComplexOutputSchema,
        )

        prompt1.push(
            messages=messages,
            output_type=OutputType.SCHEMA,
            output_schema=VeryComplexSchema,
            tools=[tool],
        )

        prompt1.pull(write_to_cache=False)
        version = prompt1.version

        prompt2 = Prompt(alias=ALIAS)
        prompt2.pull(version=version)

        # Verify output schema
        assert prompt2.output_schema is not None
        assert set(prompt2.output_schema.model_fields.keys()) == set(
            prompt1.output_schema.model_fields.keys()
        )

        # Verify tools
        assert prompt2.tools is not None
        assert len(prompt2.tools) == 1
        assert prompt2.tools[0].name == "CachedListTool"

        # Verify tool schema structure
        schema1 = prompt1.tools[0].input_schema
        schema2 = prompt2.tools[0].input_schema
        assert set(schema1["properties"].keys()) == set(
            schema2["properties"].keys()
        )

    def test_add_and_remove_tools(self):
        """Test adding and removing tools via update"""
        ALIAS = "test_prompt_list_add_remove_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"Tool management {UUID}"),
        ]

        # Start with one tool
        tool1 = Tool(
            name="InitialTool",
            description="First tool",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt.push(messages=messages, tools=[tool1])
        prompt.pull()

        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == "InitialTool"

        # Add second tool
        tool2 = Tool(
            name="SecondTool",
            description="Additional tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=ToolInputSchema,
        )

        prompt.update(
            version="latest",
            tools=[tool1, tool2],
        )
        prompt.pull()

        assert len(prompt.tools) == 2
        tool_names = {tool.name for tool in prompt.tools}
        assert tool_names == {"InitialTool", "SecondTool"}

        # Replace with just one different tool
        tool3 = Tool(
            name="ReplacementTool",
            description="Replacement",
            mode=ToolMode.ALLOW_ADDITIONAL,
            structured_schema=ComplexOutputSchema,
        )

        prompt.update(
            version="latest",
            tools=[tool3],
        )
        prompt.tools = None
        prompt.pull()

        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == "ReplacementTool"

    def test_tool_with_all_field_types(self):
        """Test tool schema with all supported field types"""
        ALIAS = "test_prompt_list_all_field_types"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        messages = [
            PromptMessage(role="user", content=f"All types test {UUID}"),
        ]

        tool = Tool(
            name="AllTypesTool",
            description="Tool with all field types",
            mode=ToolMode.STRICT,
            structured_schema=VeryComplexSchema,
        )

        prompt.push(messages=messages, tools=[tool])
        prompt.tools = []
        prompt.pull()

        input_schema = prompt.tools[0].input_schema
        props = input_schema["properties"]

        # Verify all field types are correctly represented
        assert props["id"]["type"] == "string"
        assert props["simple_field"]["type"] == "string"
        assert props["number_field"]["type"] == "integer"
        assert props["float_field"]["type"] == "number"
        assert props["bool_field"]["type"] == "boolean"
        assert props["nested_obj"]["type"] == "object"
