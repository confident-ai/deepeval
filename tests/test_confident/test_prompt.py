import pytest
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

        prompt.pull(refresh=0)

        assert prompt.hash is not None
        assert prompt.text_template == TEXT
        assert prompt.messages_template is None
        assert prompt._prompt_id is not None
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

        prompt.pull(refresh=0)

        assert prompt.hash is not None
        assert prompt.text_template == TEXT
        assert prompt.messages_template is None
        assert prompt._prompt_id is not None
        assert prompt.type == PromptType.TEXT
        assert prompt.interpolation_type == PromptInterpolationType.MUSTACHE

    def test_pull_by_hash_latest(self):
        unique_alias = f"{self.ALIAS}_{uuid.uuid4().hex[:8]}"
        prompt = Prompt(alias=unique_alias)
        UUID = uuid.uuid4()

        prompt.push(text=f"Latest content {UUID}")
        latest_hash = prompt.hash

        prompt2 = Prompt(alias=unique_alias)
        prompt2.pull(default_to_cache=False)

        assert prompt2.hash == latest_hash
        assert prompt2.text_template == f"Latest content {UUID}"

    def test_pull_by_hash_specific(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID1 = uuid.uuid4()
        prompt.push(text=f"Version 1 {UUID1}")
        hash1 = prompt.hash

        UUID2 = uuid.uuid4()
        prompt.push(text=f"Version 2 {UUID2}")

        prompt2 = Prompt(alias=self.ALIAS)
        prompt2.pull(hash=hash1)

        assert prompt2.hash == hash1
        assert prompt2.text_template == f"Version 1 {UUID1}"

    def test_pull_by_label(self):
        """Test pulling text prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt.text_template is not None
        assert prompt.type == PromptType.TEXT
        assert prompt._prompt_id is not None
        assert prompt.interpolation_type is not None

    def test_get_versions(self):
        """Test get versions for text prompt"""
        prompt = Prompt(alias=self.ALIAS)

        versions = prompt._get_versions()
        assert versions is not None

    def test_get_commits(self):
        """Test get commits for text prompt"""
        prompt = Prompt(alias=self.ALIAS)

        commits = prompt._get_commits()
        assert commits is not None

    def test_version_vs_label_vs_hash_pull(self):
        """Test that version and label pulls work independently"""

        # Pull by hash (latest)
        prompt_by_hash = Prompt(alias=self.ALIAS)
        prompt_by_hash.pull()

        # Pull by version
        prompt_by_version = Prompt(alias=self.ALIAS)
        prompt_by_version.pull(version="latest")

        # Pull by label
        prompt_by_label = Prompt(alias=self.ALIAS)
        prompt_by_label.pull(label=self.LABEL)

        # Version pull should not have label and version
        assert prompt_by_hash.hash is not None
        assert prompt_by_hash.label is None
        assert prompt_by_hash._version is None

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
        # FIX: Use a unique alias to ensure clean state
        unique_alias = f"{self.ALIAS}_cache_{uuid.uuid4().hex[:8]}"
        
        # First, ensure the prompt exists on the backend to be cached
        prompt_setup = Prompt(alias=unique_alias)
        prompt_setup.push(text=f"Setup cache content {uuid.uuid4()}")

        # Now pull and write to cache
        prompt1 = Prompt(alias=unique_alias)
        prompt1.pull(write_to_cache=True, default_to_cache=False)
        version = prompt1.version
        content = prompt1.text_template

        # Mock the API to verify no request is made
        with patch("deepeval.prompt.prompt.Api") as mock_api:
            prompt2 = Prompt(alias=unique_alias)
            prompt2.pull(version=version, default_to_cache=True)

            # Verify content matches without API call
            assert prompt2.text_template == content
            assert prompt2.version == version
            mock_api.assert_not_called()

    def test_version_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(refresh=2)

            time.sleep(5)  # polls twice in 5 seconds

            assert (
                spy_api.call_count >= 2
            )  # At least 1 polling happens after the pull
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert (
                spy_api.call_count >= 2
            )  # At least 1 polling happens after the pull
            prompt._stop_polling()

    def test_push_with_simple_output_schema(self):
        ALIAS = "test_prompt_text_simple_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Generate data {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=SimpleSchema,
        )

        prompt.pull(refresh=0)

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
        ALIAS = "test_prompt_text_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

        prompt.push(
            text=f"Generate complex data {UUID}",
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
        )

        prompt.pull(refresh=0)

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

        prompt.pull(refresh=0)

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
        TOOL_NAME = f"SearchTool_{UUID}"

        tool = Tool(
            name=TOOL_NAME,
            description="A tool for searching",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            text=f"Use the search tool {UUID}",
            tools=[tool],
        )
        prompt.tools = None
        prompt.pull(refresh=0)

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        pulled_tool = prompt.tools[0]
        assert pulled_tool.name == TOOL_NAME
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
        TOOL_NAME_1 = f"SearchTool_{UUID}"
        TOOL_NAME_2 = f"AnalysisTool_{UUID}"

        tool1 = Tool(
            name=TOOL_NAME_1,
            description="Search tool",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        tool2 = Tool(
            name=TOOL_NAME_2,
            description="Analysis tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            text=f"Use multiple tools {UUID}",
            tools=[tool1, tool2],
        )

        prompt.pull(refresh=0)

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 2

        tool_names = {tool.name for tool in prompt.tools}
        assert tool_names == {TOOL_NAME_1, TOOL_NAME_2}

        # Verify each tool
        for tool in prompt.tools:
            assert tool.structured_schema is not None
            assert tool.input_schema is not None

    def test_exiting_tool_throws_error(self):
        """Test updating a tool with the same name (now succeeds instead of throwing)"""
        ALIAS = f"test_prompt_text_update_tool_{uuid.uuid4().hex[:8]}"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()

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

        original_tool = Tool(
            name="SearchTool",
            description="Original search tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=ToolInputSchema,
        )
        
        with pytest.raises(Exception):
            prompt.push(
                text=f"Initial tool push {UUID}",
                tools=[original_tool],
            )

    def test_push_output_schema_and_tools(self):
        """Test pushing both output schema and tools together"""
        ALIAS = "test_prompt_text_schema_and_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"DataTool_{UUID}"

        tool = Tool(
            name=TOOL_NAME,
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
        prompt.pull(refresh=0)

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        assert "title" in prompt.output_schema.model_fields

        # Verify tool
        assert prompt.tools is not None
        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == TOOL_NAME

    def test_pull_preserves_tool_details(self):
        """Test that pulling preserves all tool details including schema structure"""
        ALIAS = "test_prompt_text_tool_preservation"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"DetailedTool_{UUID}"

        tool = Tool(
            name=TOOL_NAME,
            description="A tool with detailed schema",
            mode=ToolMode.STRICT,
            structured_schema=VeryComplexSchema,
        )

        prompt.push(
            text=f"Detailed tool test {UUID}",
            tools=[tool],
        )

        prompt.tools = None
        prompt.pull(refresh=0)

        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        pulled_tool = prompt.tools[0]
        assert pulled_tool.name == TOOL_NAME
        assert pulled_tool.description == "A tool with detailed schema"
        assert pulled_tool.mode == ToolMode.STRICT

        # Verify input schema has all fields
        input_schema = pulled_tool.input_schema
        assert "id" in input_schema["properties"]
        assert "simple_field" in input_schema["properties"]
        assert "nested_obj" in input_schema["properties"]

        # Verify nested structure
        nested_props = input_schema["properties"]["nested_obj"]["properties"]
        assert "level2_field" in nested_props
        assert "deep_object" in nested_props

    def test_cache_preserves_output_schema_and_tools(self):
        """Test that caching preserves output schema and tools"""
        ALIAS = "test_prompt_text_cache_schema_tools"
        prompt1 = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"CachedTool_{UUID}"

        tool = Tool(
            name=TOOL_NAME,
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
        prompt1.pull()
        hash = prompt1.hash

        # Load from cache
        prompt2 = Prompt(alias=ALIAS)
        prompt2.pull(hash=hash)

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
    LABEL = "STAGING"
    LABEL_VERSION = "00.07.01"

    def test_push(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID = str(uuid.uuid4())

        MESSAGES = [PromptMessage(role="user", content=f"Hello, world! {UUID}")]

        # generate uuid
        prompt.push(messages=MESSAGES)

        prompt.pull(refresh=0)

        assert prompt.hash is not None
        assert prompt.text_template is None
        assert prompt.messages_template == MESSAGES
        assert prompt._prompt_id is not None
        assert prompt.type == PromptType.LIST
        assert prompt.interpolation_type == PromptInterpolationType.FSTRING

    def test_push_with_interpolation_type(self):
        unique_alias = f"{self.ALIAS_WITH_INTERPOLATION_TYPE}_{uuid.uuid4().hex[:8]}"
        prompt = Prompt(alias=unique_alias)

        UUID = str(uuid.uuid4())
        MESSAGES = [PromptMessage(role="user", content=f"Hello, world! {UUID}")]

        prompt.push(
            messages=MESSAGES,
            interpolation_type=PromptInterpolationType.MUSTACHE,
        )

        # FIX: Bypass cache to assert the newly pushed interpolation type
        prompt.pull(refresh=0, default_to_cache=False)

        assert prompt.hash is not None
        assert prompt.text_template is None
        assert prompt.messages_template == MESSAGES
        assert prompt._prompt_id is not None
        assert prompt.type == PromptType.LIST
        assert prompt.interpolation_type == PromptInterpolationType.MUSTACHE

    def test_pull_by_hash_latest(self):
        unique_alias = f"{self.ALIAS}_{uuid.uuid4().hex[:8]}"
        prompt = Prompt(alias=unique_alias)
        UUID = uuid.uuid4()

        MESSAGES = [
            PromptMessage(role="user", content=f"Latest content {UUID}")
        ]
        prompt.push(messages=MESSAGES)
        latest_hash = prompt.hash

        prompt2 = Prompt(alias=unique_alias)
        # FIX: Bypass cache
        prompt2.pull(default_to_cache=False)

        assert prompt2.hash == latest_hash
        assert prompt2.messages_template == MESSAGES

    def test_pull_by_hash_specific(self):
        prompt = Prompt(alias=self.ALIAS)

        UUID1 = uuid.uuid4()
        MESSAGES1 = [PromptMessage(role="user", content=f"Version 1 {UUID1}")]
        prompt.push(messages=MESSAGES1)
        hash1 = prompt.hash

        UUID2 = uuid.uuid4()
        MESSAGES2 = [PromptMessage(role="user", content=f"Version 2 {UUID2}")]
        prompt.push(messages=MESSAGES2)

        prompt2 = Prompt(alias=self.ALIAS)
        prompt2.pull(hash=hash1)

        assert prompt2.hash == hash1
        assert prompt2.messages_template == MESSAGES1

    def test_pull_by_label(self):
        """Test pulling list prompt by label"""
        prompt = Prompt(alias=self.ALIAS)

        # Pull by label
        prompt.pull(label=self.LABEL)

        assert prompt.label == self.LABEL
        assert prompt.version == self.LABEL_VERSION
        assert prompt.messages_template is not None
        assert prompt.type == PromptType.LIST
        assert prompt._prompt_id is not None
        assert prompt.interpolation_type is not None

    def test_get_versions(self):
        """Test get versions for list prompt"""
        prompt = Prompt(alias=self.ALIAS)

        versions = prompt._get_versions()
        assert versions is not None

    def test_get_commits(self):
        """Test get commits for list prompt"""
        prompt = Prompt(alias=self.ALIAS)

        commits = prompt._get_commits()
        assert commits is not None

    def test_version_vs_label_vs_hash_pull(self):
        """Test that version and label pulls work independently"""

        # Pull by hash (latest)
        prompt_by_hash = Prompt(alias=self.ALIAS)
        prompt_by_hash.pull()

        # Pull by version
        prompt_by_version = Prompt(alias=self.ALIAS)
        prompt_by_version.pull(version="latest")

        # Pull by label
        prompt_by_label = Prompt(alias=self.ALIAS)
        prompt_by_label.pull(label=self.LABEL)

        # Version pull should not have label and version
        assert prompt_by_hash.hash is not None
        assert prompt_by_hash.label is None
        assert prompt_by_hash._version is None

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
            prompt.pull(refresh=2)

            time.sleep(5)  # polls twice in 5 seconds

            assert (
                spy_api.call_count >= 2
            )  # At least 1 polling happens after the pull
            prompt._stop_polling()

    def test_label_polling(self):
        # Use wraps to spy on real API calls while still counting them
        with patch("deepeval.prompt.prompt.Api", wraps=Api) as spy_api:
            prompt = Prompt(alias=self.ALIAS)
            prompt.pull(label=self.LABEL, refresh=2)

            time.sleep(5)  # polls twice in 5 seconds

            assert prompt.version == self.LABEL_VERSION
            assert (
                spy_api.call_count >= 2
            )  # At least 1 polling happens after the pull
            prompt._stop_polling()

    def test_push_with_simple_output_schema(self):
        ALIAS = "test_prompt_list_simple_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        MESSAGES = [PromptMessage(role="user", content=f"Generate data {UUID}")]

        prompt.push(
            messages=MESSAGES,
            output_type=OutputType.SCHEMA,
            output_schema=SimpleSchema,
        )

        prompt.pull(refresh=0)

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
        ALIAS = "test_prompt_list_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        MESSAGES = [
            PromptMessage(role="user", content=f"Generate complex data {UUID}")
        ]

        prompt.push(
            messages=MESSAGES,
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
        )

        prompt.pull(refresh=0)

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
        """Test pushing list prompt with deeply nested output schema (3 levels)"""
        ALIAS = "test_prompt_list_deep_nested_schema"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        MESSAGES = [
            PromptMessage(
                role="user", content=f"Generate very complex data {UUID}"
            )
        ]

        prompt.push(
            messages=MESSAGES,
            output_type=OutputType.SCHEMA,
            output_schema=VeryComplexSchema,
        )

        prompt.pull(refresh=0)

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
        """Test pushing list prompt with a single tool"""
        ALIAS = "test_prompt_list_single_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"SearchTool_{UUID}"
        MESSAGES = [
            PromptMessage(role="user", content=f"Use the search tool {UUID}")
        ]

        tool = Tool(
            name=TOOL_NAME,
            description="A tool for searching",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            messages=MESSAGES,
            tools=[tool],
        )
        prompt.tools = None
        prompt.pull(refresh=0)

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        pulled_tool = prompt.tools[0]
        assert pulled_tool.name == TOOL_NAME
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
        """Test pushing list prompt with multiple tools"""
        ALIAS = "test_prompt_list_multiple_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME_1 = f"SearchTool_{UUID}"
        TOOL_NAME_2 = f"AnalysisTool_{UUID}"
        MESSAGES = [
            PromptMessage(role="user", content=f"Use multiple tools {UUID}")
        ]

        tool1 = Tool(
            name=TOOL_NAME_1,
            description="Search tool",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        tool2 = Tool(
            name=TOOL_NAME_2,
            description="Analysis tool",
            mode=ToolMode.NO_ADDITIONAL,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            messages=MESSAGES,
            tools=[tool1, tool2],
        )

        prompt.pull(refresh=0)

        # Verify tools
        assert prompt.tools is not None
        assert len(prompt.tools) == 2

        tool_names = {tool.name for tool in prompt.tools}
        assert tool_names == {TOOL_NAME_1, TOOL_NAME_2}

        # Verify each tool
        for tool in prompt.tools:
            assert tool.structured_schema is not None
            assert tool.input_schema is not None

    def test_exiting_tool_throws_error(self):
        """Test updating a tool with the same name (should replace it)"""
        ALIAS = "test_prompt_list_update_tool"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        MESSAGES = [
            PromptMessage(role="user", content=f"Initial tool push {UUID}")
        ]

        # Push initial tool
        original_tool = Tool(
            name="SearchTool",
            description="Original search tool",
            mode=ToolMode.STRICT,
            structured_schema=ToolInputSchema,
        )

        prompt.push(
            messages=MESSAGES,
            tools=[original_tool],
        )

        original_tool = Tool(
            name="SearchTool",
            description="Original search tool",
            mode=ToolMode.ALLOW_ADDITIONAL,
            structured_schema=ToolInputSchema,
        )

        with pytest.raises(Exception):
            prompt.push(
                messages=MESSAGES,
                tools=[original_tool],
            )

    def test_push_output_schema_and_tools(self):
        """Test pushing both output schema and tools together"""
        ALIAS = "test_prompt_list_schema_and_tools"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"DataTool_{UUID}"
        MESSAGES = [
            PromptMessage(role="user", content=f"Process data with tool {UUID}")
        ]

        tool = Tool(
            name=TOOL_NAME,
            description="Data processing tool",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt.push(
            messages=MESSAGES,
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
            tools=[tool],
        )
        prompt.output_schema = None
        prompt.tools = None
        prompt.pull(refresh=0)

        # Verify output schema
        assert prompt.output_type == OutputType.SCHEMA
        assert prompt.output_schema is not None
        assert "title" in prompt.output_schema.model_fields

        # Verify tool
        assert prompt.tools is not None
        assert len(prompt.tools) == 1
        assert prompt.tools[0].name == TOOL_NAME

    def test_pull_preserves_tool_details(self):
        """Test that pulling preserves all tool details including schema structure"""
        ALIAS = "test_prompt_list_tool_preservation"
        prompt = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"DetailedTool_{UUID}"
        MESSAGES = [
            PromptMessage(role="user", content=f"Detailed tool test {UUID}")
        ]

        tool = Tool(
            name=TOOL_NAME,
            description="A tool with detailed schema",
            mode=ToolMode.STRICT,
            structured_schema=VeryComplexSchema,
        )

        prompt.push(
            messages=MESSAGES,
            tools=[tool],
        )

        prompt.tools = None
        prompt.pull(refresh=0)

        assert prompt.tools is not None
        assert len(prompt.tools) == 1

        pulled_tool = prompt.tools[0]
        assert pulled_tool.name == TOOL_NAME
        assert pulled_tool.description == "A tool with detailed schema"
        assert pulled_tool.mode == ToolMode.STRICT

        # Verify input schema has all fields
        input_schema = pulled_tool.input_schema
        assert "id" in input_schema["properties"]
        assert "simple_field" in input_schema["properties"]
        assert "nested_obj" in input_schema["properties"]

        # Verify nested structure
        nested_props = input_schema["properties"]["nested_obj"]["properties"]
        assert "level2_field" in nested_props
        assert "deep_object" in nested_props

    def test_cache_preserves_output_schema_and_tools(self):
        """Test that caching preserves output schema and tools"""
        ALIAS = "test_prompt_list_cache_schema_tools"
        prompt1 = Prompt(alias=ALIAS)

        UUID = uuid.uuid4()
        TOOL_NAME = f"CachedTool_{UUID}"
        MESSAGES = [PromptMessage(role="user", content=f"Cache test {UUID}")]

        tool = Tool(
            name=TOOL_NAME,
            description="Tool for cache test",
            mode=ToolMode.STRICT,
            structured_schema=SimpleSchema,
        )

        prompt1.push(
            messages=MESSAGES,
            output_type=OutputType.SCHEMA,
            output_schema=ComplexOutputSchema,
            tools=[tool],
        )

        # Pull and cache
        prompt1.pull()
        hash = prompt1.hash

        # Load from cache
        prompt2 = Prompt(alias=ALIAS)
        prompt2.pull(hash=hash)

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
