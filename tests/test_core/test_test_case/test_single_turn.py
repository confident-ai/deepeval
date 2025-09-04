import pytest
import uuid
from unittest.mock import patch
from pydantic import ValidationError

from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    LLMTestCaseParams,
    ToolCallParams,
)
from deepeval.test_case.mcp import MCPServer


class TestLLMTestCaseInitialization:

    def test_minimal_initialization(self):
        test_case = LLMTestCase(input="What is the capital of France?")

        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output is None
        assert test_case.expected_output is None
        assert test_case.context is None
        assert test_case.retrieval_context is None
        assert test_case.additional_metadata is None
        assert test_case.tools_called is None
        assert test_case.comments is None
        assert test_case.expected_tools is None
        assert test_case.token_cost is None
        assert test_case.completion_time is None
        assert test_case.name is None
        assert test_case.tags is None
        assert test_case.mcp_servers is None
        assert test_case.mcp_tools_called is None
        assert test_case.mcp_resources_called is None
        assert test_case.mcp_prompts_called is None

        # Test private attributes have defaults
        assert test_case._trace_dict is None
        assert test_case._dataset_rank is None
        assert test_case._dataset_alias is None
        assert test_case._dataset_id is None
        assert isinstance(test_case._identifier, str)

    def test_full_initialization(self):
        tool_call = ToolCall(
            name="search_tool",
            description="A search tool",
            reasoning="Need to search for information",
            output={"results": ["result1", "result2"]},
            input_parameters={"query": "test query"},
        )

        test_case = LLMTestCase(
            input="What is machine learning?",
            actual_output="Machine learning is a subset of AI...",
            expected_output="Machine learning is a method of data analysis...",
            context=["ML is important", "AI revolution"],
            retrieval_context=["Retrieved context 1", "Retrieved context 2"],
            additional_metadata={"source": "test", "version": 1.0},
            tools_called=[tool_call],
            comments="This is a test case",
            expected_tools=[tool_call],
            token_cost=0.05,
            completion_time=1.25,
            name="ML Question Test",
            tags=["machine-learning", "AI", "test"],
        )

        assert test_case.input == "What is machine learning?"
        assert (
            test_case.actual_output == "Machine learning is a subset of AI..."
        )
        assert (
            test_case.expected_output
            == "Machine learning is a method of data analysis..."
        )
        assert test_case.context == ["ML is important", "AI revolution"]
        assert test_case.retrieval_context == [
            "Retrieved context 1",
            "Retrieved context 2",
        ]
        assert test_case.additional_metadata == {
            "source": "test",
            "version": 1.0,
        }
        assert len(test_case.tools_called) == 1
        assert test_case.tools_called[0] == tool_call
        assert test_case.comments == "This is a test case"
        assert len(test_case.expected_tools) == 1
        assert test_case.expected_tools[0] == tool_call
        assert test_case.token_cost == 0.05
        assert test_case.completion_time == 1.25
        assert test_case.name == "ML Question Test"
        assert test_case.tags == ["machine-learning", "AI", "test"]


class TestLLMTestCaseCamelCaseInitialization:

    def test_camelcase_field_initialization(self):
        input_text = "What is artificial intelligence?"
        actual_output_text = "AI is a branch of computer science..."
        expected_output_text = "AI involves creating smart machines..."
        context_list = ["AI is important", "Technology revolution"]
        retrieval_context_list = [
            "Retrieved AI context 1",
            "Retrieved AI context 2",
        ]
        metadata_dict = {"source": "camelCase test", "version": 2.0}
        comments_text = "This is a camelCase test case"
        token_cost_value = 0.08
        completion_time_value = 2.5
        name_text = "AI Question Test CamelCase"
        tags_list = ["artificial-intelligence", "camelCase", "test"]

        tool_call = ToolCall(
            name="search_tool",
            description="A search tool",
            reasoning="Need to search for information",
            output={"results": ["result1", "result2"]},
            inputParameters={
                "query": "test query"
            },  # camelCase for input_parameters
        )

        test_case = LLMTestCase(
            input=input_text,
            actualOutput=actual_output_text,  # camelCase
            expectedOutput=expected_output_text,  # camelCase
            context=context_list,
            retrievalContext=retrieval_context_list,  # camelCase
            additionalMetadata=metadata_dict,  # camelCase
            toolsCalled=[tool_call],  # camelCase
            comments=comments_text,
            expectedTools=[tool_call],  # camelCase
            tokenCost=token_cost_value,  # camelCase
            completionTime=completion_time_value,  # camelCase
            name=name_text,
            tags=tags_list,
        )

        # Verify all fields are properly set using the same variables
        assert test_case.input == input_text
        assert test_case.actual_output == actual_output_text
        assert test_case.expected_output == expected_output_text
        assert test_case.context == context_list
        assert test_case.retrieval_context == retrieval_context_list
        assert test_case.additional_metadata == metadata_dict
        assert len(test_case.tools_called) == 1
        assert test_case.tools_called[0] == tool_call
        assert test_case.comments == comments_text
        assert len(test_case.expected_tools) == 1
        assert test_case.expected_tools[0] == tool_call
        assert test_case.token_cost == token_cost_value
        assert test_case.completion_time == completion_time_value
        assert test_case.name == name_text
        assert test_case.tags == tags_list

    def test_mixed_case_initialization(self):
        input_text = "Mixed case test"
        actual_output_text = "This uses camelCase"
        expected_output_text = "This uses snake_case"
        context_list = ["mixed", "case"]
        retrieval_context_list = ["snake_case context"]
        metadata_dict = {"mixed": "case"}
        token_cost_value = 0.02
        completion_time_value = 1.0

        test_case = LLMTestCase(
            input=input_text,
            actualOutput=actual_output_text,  # camelCase
            expected_output=expected_output_text,  # snake_case
            context=context_list,
            retrieval_context=retrieval_context_list,  # snake_case
            additionalMetadata=metadata_dict,  # camelCase
            token_cost=token_cost_value,  # snake_case
            completionTime=completion_time_value,  # camelCase
        )

        assert test_case.input == input_text
        assert test_case.actual_output == actual_output_text
        assert test_case.expected_output == expected_output_text
        assert test_case.context == context_list
        assert test_case.retrieval_context == retrieval_context_list
        assert test_case.additional_metadata == metadata_dict
        assert test_case.token_cost == token_cost_value
        assert test_case.completion_time == completion_time_value

    def test_tool_call_camelcase_initialization(self):
        # Test data variables
        input_text = "Tool parameter test"
        camel_tool_name = "camel_tool"
        camel_tool_description = "A tool with camelCase params"
        camel_tool_reasoning = "Testing camelCase"
        camel_input_params = {"queryParam": "camelCase value", "maxResults": 10}
        camel_output = {"camelCaseResult": "success"}

        snake_tool_name = "snake_tool"
        snake_tool_description = "A tool with snake_case params"
        snake_tool_reasoning = "Testing snake_case"
        snake_input_params = {
            "query_param": "snake_case value",
            "max_results": 5,
        }
        snake_output = {"snake_case_result": "success"}

        # Test ToolCall with camelCase
        tool_call_camel = ToolCall(
            name=camel_tool_name,
            description=camel_tool_description,
            reasoning=camel_tool_reasoning,
            inputParameters=camel_input_params,
            output=camel_output,
        )

        # Test ToolCall with snake_case
        tool_call_snake = ToolCall(
            name=snake_tool_name,
            description=snake_tool_description,
            reasoning=snake_tool_reasoning,
            input_parameters=snake_input_params,
            output=snake_output,
        )

        test_case = LLMTestCase(
            input=input_text,
            toolsCalled=[tool_call_camel, tool_call_snake],
        )

        assert len(test_case.tools_called) == 2
        assert test_case.tools_called[0].name == camel_tool_name
        assert test_case.tools_called[0].input_parameters == camel_input_params
        assert test_case.tools_called[1].name == snake_tool_name
        assert test_case.tools_called[1].input_parameters == snake_input_params


class TestLLMTestCaseTypeValidation:

    def test_input_must_be_string(self):
        with pytest.raises(TypeError, match="'input' must be a string"):
            LLMTestCase(input=123)

        with pytest.raises(TypeError, match="'input' must be a string"):
            LLMTestCase(input=["not", "a", "string"])

        with pytest.raises(ValidationError):
            LLMTestCase(input=None)

    def test_actual_output_must_be_string_or_none(self):
        """Test that actual_output must be a string or None."""
        # Valid cases
        LLMTestCase(input="test", actual_output="valid output")
        LLMTestCase(input="test", actual_output=None)

        # Invalid cases
        with pytest.raises(TypeError, match="'actual_output' must be a string"):
            LLMTestCase(input="test", actual_output=123)

        with pytest.raises(TypeError, match="'actual_output' must be a string"):
            LLMTestCase(input="test", actual_output=["not", "string"])

    def test_context_must_be_list_of_strings_or_none(self):
        """Test that context must be None or list of strings."""
        # Valid cases
        LLMTestCase(input="test", context=None)
        LLMTestCase(input="test", context=[])
        LLMTestCase(input="test", context=["context1", "context2"])

        # Invalid cases - not a list
        with pytest.raises(
            TypeError, match="'context' must be None or a list of strings"
        ):
            LLMTestCase(input="test", context="not a list")

        # Invalid cases - list with non-strings
        with pytest.raises(
            TypeError, match="'context' must be None or a list of strings"
        ):
            LLMTestCase(input="test", context=["valid", 123, "mixed"])

        with pytest.raises(
            TypeError, match="'context' must be None or a list of strings"
        ):
            LLMTestCase(input="test", context=[None, "string"])

    def test_retrieval_context_must_be_list_of_strings_or_none(self):
        """Test that retrieval_context must be None or list of strings."""
        # Valid cases
        LLMTestCase(input="test", retrieval_context=None)
        LLMTestCase(input="test", retrieval_context=[])
        LLMTestCase(input="test", retrieval_context=["context1", "context2"])

        # Invalid cases
        with pytest.raises(
            TypeError,
            match="'retrieval_context' must be None or a list of strings",
        ):
            LLMTestCase(input="test", retrieval_context="not a list")

        with pytest.raises(
            TypeError,
            match="'retrieval_context' must be None or a list of strings",
        ):
            LLMTestCase(input="test", retrieval_context=["valid", 123])

    def test_tools_called_must_be_list_of_toolcall_or_none(self):
        """Test that tools_called must be None or list of ToolCall objects."""
        tool_call = ToolCall(name="test_tool")

        # Valid cases
        LLMTestCase(input="test", tools_called=None)
        LLMTestCase(input="test", tools_called=[])
        LLMTestCase(input="test", tools_called=[tool_call])

        # Invalid cases - not a list
        with pytest.raises(
            TypeError,
            match="'tools_called' must be None or a list of `ToolCall`",
        ):
            LLMTestCase(input="test", tools_called="not a list")

        # Invalid cases - list with non-ToolCall objects
        with pytest.raises(
            TypeError,
            match="'tools_called' must be None or a list of `ToolCall`",
        ):
            LLMTestCase(
                input="test", tools_called=[tool_call, "not a toolcall"]
            )

        with pytest.raises(
            TypeError,
            match="'tools_called' must be None or a list of `ToolCall`",
        ):
            LLMTestCase(input="test", tools_called=[{"name": "dict_tool"}])

    def test_expected_tools_must_be_list_of_toolcall_or_none(self):
        """Test that expected_tools must be None or list of ToolCall objects."""
        tool_call = ToolCall(name="test_tool")

        # Valid cases
        LLMTestCase(input="test", expected_tools=None)
        LLMTestCase(input="test", expected_tools=[])
        LLMTestCase(input="test", expected_tools=[tool_call])

        # Invalid cases
        with pytest.raises(
            TypeError,
            match="'expected_tools' must be None or a list of `ToolCall`",
        ):
            LLMTestCase(
                input="test", expected_tools=[tool_call, "not a toolcall"]
            )


class TestToolCallFunctionality:
    """Test ToolCall class functionality."""

    def test_tool_call_minimal_initialization(self):
        """Test ToolCall with only required field."""
        tool_call = ToolCall(name="test_tool")

        assert tool_call.name == "test_tool"
        assert tool_call.description is None
        assert tool_call.reasoning is None
        assert tool_call.output is None
        assert tool_call.input_parameters is None

    def test_tool_call_full_initialization(self):
        """Test ToolCall with all fields."""
        tool_call = ToolCall(
            name="search_tool",
            description="Searches the web",
            reasoning="User needs current information",
            output={"results": ["result1", "result2"], "count": 2},
            input_parameters={"query": "latest news", "limit": 10},
        )

        assert tool_call.name == "search_tool"
        assert tool_call.description == "Searches the web"
        assert tool_call.reasoning == "User needs current information"
        assert tool_call.output == {
            "results": ["result1", "result2"],
            "count": 2,
        }
        assert tool_call.input_parameters == {
            "query": "latest news",
            "limit": 10,
        }

    def test_tool_call_equality(self):
        """Test ToolCall equality comparison."""
        tool1 = ToolCall(
            name="test_tool",
            input_parameters={"param": "value"},
            output="result",
        )
        tool2 = ToolCall(
            name="test_tool",
            input_parameters={"param": "value"},
            output="result",
        )
        tool3 = ToolCall(
            name="different_tool",
            input_parameters={"param": "value"},
            output="result",
        )

        assert tool1 == tool2
        assert tool1 != tool3
        assert tool1 != "not a toolcall"

    def test_tool_call_hashing(self):
        """Test ToolCall hashing functionality."""
        tool1 = ToolCall(
            name="test_tool",
            input_parameters={"param": "value", "nested": {"key": "val"}},
            output={"result": ["item1", "item2"]},
        )
        tool2 = ToolCall(
            name="test_tool",
            input_parameters={"param": "value", "nested": {"key": "val"}},
            output={"result": ["item1", "item2"]},
        )

        # Same tools should have same hash
        assert hash(tool1) == hash(tool2)

        # Different tools should have different hash (with high probability)
        tool3 = ToolCall(name="different_tool")
        assert hash(tool1) != hash(tool3)

        # Test that tools can be used in sets
        tool_set = {tool1, tool2, tool3}
        assert len(tool_set) == 2  # tool1 and tool2 are equal

    def test_tool_call_hashing_with_complex_types(self):
        tool_call = ToolCall(
            name="complex_tool",
            input_parameters={
                "list_param": [1, 2, {"nested": "dict"}],
                "dict_param": {"key": [1, 2, 3]},
                "none_param": None,
            },
            output=[{"complex": "output"}, ["with", "lists"]],
        )

        # Should not raise an error
        hash_value = hash(tool_call)
        assert isinstance(hash_value, int)

    def test_tool_call_repr(self):
        tool_call = ToolCall(
            name="test_tool",
            description="A test tool",
            reasoning="For testing",
            input_parameters={"query": "test"},
            output={"result": "success"},
        )

        repr_str = repr(tool_call)
        assert "ToolCall(" in repr_str
        assert 'name="test_tool"' in repr_str
        assert 'description="A test tool"' in repr_str
        assert 'reasoning="For testing"' in repr_str
        assert "input_parameters=" in repr_str
        assert "output=" in repr_str

    def test_tool_call_repr_minimal(self):
        tool_call = ToolCall(name="minimal_tool")
        repr_str = repr(tool_call)

        assert "ToolCall(" in repr_str
        assert 'name="minimal_tool"' in repr_str
        assert "description=" not in repr_str
        assert "reasoning=" not in repr_str


class TestEdgeCases:

    def test_empty_strings(self):
        test_case = LLMTestCase(
            input="", actual_output="", expected_output="", comments=""
        )

        assert test_case.input == ""
        assert test_case.actual_output == ""
        assert test_case.expected_output == ""
        assert test_case.comments == ""

    def test_empty_lists(self):
        test_case = LLMTestCase(
            input="test",
            context=[],
            retrieval_context=[],
            tools_called=[],
            expected_tools=[],
            tags=[],
        )

        assert test_case.context == []
        assert test_case.retrieval_context == []
        assert test_case.tools_called == []
        assert test_case.expected_tools == []
        assert test_case.tags == []

    def test_very_long_strings(self):
        long_string = "a" * 10000

        test_case = LLMTestCase(
            input=long_string,
            actual_output=long_string,
            expected_output=long_string,
        )

        assert len(test_case.input) == 10000
        assert len(test_case.actual_output) == 10000
        assert len(test_case.expected_output) == 10000

    def test_special_characters(self):
        special_input = "Hello 世界! 🌍 ñoño @#$%^&*()[]{}|\\:;\"'<>?,./"

        test_case = LLMTestCase(
            input=special_input,
            actual_output=special_input,
            context=[special_input],
        )

        assert test_case.input == special_input
        assert test_case.actual_output == special_input
        assert test_case.context[0] == special_input

    def test_numeric_edge_values(self):
        test_case = LLMTestCase(
            input="test", token_cost=0.0, completion_time=0.0
        )
        assert test_case.token_cost == 0.0
        assert test_case.completion_time == 0.0

        test_case = LLMTestCase(
            input="test", token_cost=float("inf"), completion_time=float("inf")
        )
        assert test_case.token_cost == float("inf")
        assert test_case.completion_time == float("inf")

    def test_large_lists(self):
        large_context = [f"context_item_{i}" for i in range(1000)]
        large_tools = [ToolCall(name=f"tool_{i}") for i in range(100)]

        test_case = LLMTestCase(
            input="test", context=large_context, tools_called=large_tools
        )

        assert len(test_case.context) == 1000
        assert len(test_case.tools_called) == 100
        assert test_case.context[0] == "context_item_0"
        assert test_case.tools_called[0].name == "tool_0"

    def test_deeply_nested_structures(self):
        nested_structure = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["deep", "nested", {"level5": "value"}]
                    }
                }
            }
        }

        tool_call = ToolCall(
            name="nested_tool",
            input_parameters=nested_structure,
            output=nested_structure,
        )

        test_case = LLMTestCase(input="test", tools_called=[tool_call])

        assert test_case.tools_called[0].input_parameters == nested_structure
        assert test_case.tools_called[0].output == nested_structure


class TestSerialization:

    def test_serialization_aliases(self):
        test_case = LLMTestCase(
            input="test",
            actual_output="output",
            expected_output="expected",
            context=["context"],
            retrieval_context=["retrieval"],
            tools_called=[ToolCall(name="tool")],
            expected_tools=[ToolCall(name="expected_tool")],
            token_cost=0.05,
            completion_time=1.0,
        )

        # Test model dump with aliases
        model_dict = test_case.model_dump(by_alias=True)

        assert "actualOutput" in model_dict
        assert "expectedOutput" in model_dict
        assert "context" in model_dict
        assert "retrievalContext" in model_dict
        assert "toolsCalled" in model_dict
        assert "expectedTools" in model_dict
        assert "tokenCost" in model_dict
        assert "completionTime" in model_dict

    def test_additional_metadata_serialization(self):
        metadata = {
            "source": "test",
            "timestamp": "2024-01-01",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        test_case = LLMTestCase(input="test", additional_metadata=metadata)

        assert test_case.additional_metadata == metadata

        model_dict = test_case.model_dump(by_alias=True)
        assert "additionalMetadata" in model_dict
        assert model_dict["additionalMetadata"] == metadata


class TestLLMTestCaseParams:
    def test_enum_values(self):
        assert LLMTestCaseParams.INPUT.value == "input"
        assert LLMTestCaseParams.ACTUAL_OUTPUT.value == "actual_output"
        assert LLMTestCaseParams.EXPECTED_OUTPUT.value == "expected_output"
        assert LLMTestCaseParams.CONTEXT.value == "context"
        assert LLMTestCaseParams.RETRIEVAL_CONTEXT.value == "retrieval_context"
        assert LLMTestCaseParams.TOOLS_CALLED.value == "tools_called"
        assert LLMTestCaseParams.EXPECTED_TOOLS.value == "expected_tools"
        assert LLMTestCaseParams.MCP_SERVERS.value == "mcp_servers"
        assert LLMTestCaseParams.MCP_TOOLS_CALLED.value == "mcp_tools_called"
        assert (
            LLMTestCaseParams.MCP_RESOURCES_CALLED.value
            == "mcp_resources_called"
        )
        assert (
            LLMTestCaseParams.MCP_PROMPTS_CALLED.value == "mcp_prompts_called"
        )


class TestToolCallParams:
    def test_enum_values(self):
        assert ToolCallParams.INPUT_PARAMETERS.value == "input_parameters"
        assert ToolCallParams.OUTPUT.value == "output"


class TestPrivateAttributes:
    def test_private_attributes_not_in_model_dump(self):
        test_case = LLMTestCase(input="test")

        model_dict = test_case.model_dump()

        assert "_trace_dict" not in model_dict
        assert "_dataset_rank" not in model_dict
        assert "_dataset_alias" not in model_dict
        assert "_dataset_id" not in model_dict
        assert "_identifier" not in model_dict

    def test_private_attributes_accessible(self):
        test_case = LLMTestCase(input="test")

        assert test_case._trace_dict is None
        assert test_case._dataset_rank is None
        assert test_case._dataset_alias is None
        assert test_case._dataset_id is None
        assert isinstance(test_case._identifier, str)

        test_case._trace_dict = {"key": "value"}
        test_case._dataset_rank = 1
        test_case._dataset_alias = "test_alias"
        test_case._dataset_id = "test_id"

        assert test_case._trace_dict == {"key": "value"}
        assert test_case._dataset_rank == 1
        assert test_case._dataset_alias == "test_alias"
        assert test_case._dataset_id == "test_id"

    def test_identifier_is_unique(self):
        test_case1 = LLMTestCase(input="test1")
        test_case2 = LLMTestCase(input="test2")

        assert test_case1._identifier != test_case2._identifier

        # Both should be valid UUIDs
        uuid.UUID(test_case1._identifier)  # Will raise if invalid
        uuid.UUID(test_case2._identifier)  # Will raise if invalid


class TestIntegrationScenarios:
    def test_rag_evaluation_scenario(self):
        test_case = LLMTestCase(
            input="What are the benefits of renewable energy?",
            actual_output="Renewable energy offers several benefits including environmental protection, energy independence, and economic advantages...",
            expected_output="Renewable energy provides environmental benefits by reducing greenhouse gas emissions, economic benefits through job creation, and energy security through reduced dependence on fossil fuels.",
            context=[
                "Renewable energy is crucial for fighting climate change",
                "Solar and wind power create jobs",
                "Energy independence reduces geopolitical risks",
            ],
            retrieval_context=[
                "Solar energy reduces carbon emissions by 90%",
                "Wind power industry employed 130,000 people in 2023",
                "Countries with renewable energy have better energy security",
            ],
            additional_metadata={
                "source_documents": ["doc1.pdf", "doc2.pdf"],
                "retrieval_score": 0.85,
                "model_version": "gpt-4",
            },
            token_cost=0.03,
            completion_time=2.1,
            name="Renewable Energy Benefits",
            tags=["environment", "energy", "rag"],
        )

        assert "renewable energy" in test_case.input.lower()
        assert len(test_case.context) == 3
        assert len(test_case.retrieval_context) == 3
        assert test_case.additional_metadata["retrieval_score"] == 0.85

    def test_tool_calling_scenario(self):
        search_tool = ToolCall(
            name="web_search",
            description="Search the web for current information",
            reasoning="User asked about current events, need up-to-date information",
            input_parameters={
                "query": "latest AI developments 2024",
                "max_results": 5,
            },
            output={
                "results": [
                    {
                        "title": "AI Breakthrough 2024",
                        "url": "example.com/ai",
                        "snippet": "Recent advances...",
                    },
                    {
                        "title": "ML Innovation",
                        "url": "example.com/ml",
                        "snippet": "New techniques...",
                    },
                ],
                "total_found": 127,
            },
        )

        calc_tool = ToolCall(
            name="calculator",
            description="Perform mathematical calculations",
            reasoning="Need to calculate market size based on search results",
            input_parameters={"expression": "127 * 0.15"},
            output={"result": 19.05, "formatted": "19.05"},
        )

        test_case = LLMTestCase(
            input="What are the latest AI developments and what percentage might be relevant to healthcare?",
            actual_output="Based on my search, there are 127 recent AI developments, with approximately 19 (15%) being healthcare-related...",
            tools_called=[search_tool, calc_tool],
            expected_tools=[
                search_tool
            ],  # Expected to search, calculation was bonus
            token_cost=0.08,
            completion_time=4.2,
            name="AI Developments Tool Use",
            tags=["tools", "search", "calculation", "AI"],
        )

        assert len(test_case.tools_called) == 2
        assert test_case.tools_called[0].name == "web_search"
        assert test_case.tools_called[1].name == "calculator"
        assert len(test_case.expected_tools) == 1
