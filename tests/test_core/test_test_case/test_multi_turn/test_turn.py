import pytest
from pydantic import ValidationError
from deepeval.test_case import Turn, ToolCall


class TestTurnInitialization:

    def test_minimal_initialization(self):
        turn = Turn(role="user", content="Hello")

        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.user_id is None
        assert turn.retrieval_context is None
        assert turn.tools_called is None
        assert turn.mcp_tools_called is None
        assert turn.mcp_resources_called is None
        assert turn.mcp_prompts_called is None
        assert turn.additional_metadata is None

    def test_user_role_initialization(self):
        turn = Turn(role="user", content="What's the weather like?")
        assert turn.role == "user"
        assert turn.content == "What's the weather like?"

    def test_assistant_role_initialization(self):
        turn = Turn(role="assistant", content="It's sunny today!")
        assert turn.role == "assistant"
        assert turn.content == "It's sunny today!"

    def test_full_initialization(self):
        tool_call = ToolCall(
            name="weather_tool",
            description="Get weather info",
            reasoning="User asked about weather",
            output={"temperature": "75F", "condition": "sunny"},
        )

        turn = Turn(
            role="assistant",
            content="Let me check the weather for you.",
            user_id="user123",
            retrieval_context=["Weather data from API", "Current conditions"],
            tools_called=[tool_call],
            additional_metadata={
                "timestamp": "2024-01-01T10:00:00Z",
                "model": "gpt-4",
            },
        )

        assert turn.role == "assistant"
        assert turn.content == "Let me check the weather for you."
        assert turn.user_id == "user123"
        assert len(turn.retrieval_context) == 2
        assert len(turn.tools_called) == 1
        assert turn.tools_called[0].name == "weather_tool"
        assert turn.additional_metadata["model"] == "gpt-4"


class TestTurnValidation:

    def test_invalid_role_raises_error(self):
        with pytest.raises(ValidationError):
            Turn(role="invalid_role", content="Hello")

    def test_empty_content_is_valid(self):
        turn = Turn(role="user", content="")
        assert turn.content == ""

    def test_none_content_raises_error(self):
        with pytest.raises(ValidationError):
            Turn(role="user", content=None)

    def test_non_string_content_raises_error(self):
        with pytest.raises(ValidationError):
            Turn(role="user", content=123)


class TestTurnWithRetrievalContext:

    def test_single_context_item(self):
        turn = Turn(
            role="assistant",
            content="Based on the documentation...",
            retrieval_context=["API documentation for weather service"],
        )
        assert len(turn.retrieval_context) == 1
        assert (
            turn.retrieval_context[0] == "API documentation for weather service"
        )

    def test_multiple_context_items(self):
        contexts = [
            "Weather API documentation",
            "Historical weather data",
            "User location preferences",
        ]
        turn = Turn(
            role="assistant",
            content="Weather forecast",
            retrieval_context=contexts,
        )
        assert len(turn.retrieval_context) == 3
        assert turn.retrieval_context == contexts

    def test_empty_retrieval_context_list(self):
        turn = Turn(role="assistant", content="Response", retrieval_context=[])
        assert turn.retrieval_context == []

    def test_none_retrieval_context(self):
        turn = Turn(role="assistant", content="Response")
        assert turn.retrieval_context is None


class TestTurnWithTools:

    def test_single_tool_call(self):
        tool_call = ToolCall(
            name="search_tool",
            description="Search for information",
            reasoning="User needs search results",
            output={"results": ["result1", "result2"]},
        )

        turn = Turn(
            role="assistant",
            content="Let me search for that.",
            tools_called=[tool_call],
        )

        assert len(turn.tools_called) == 1
        assert turn.tools_called[0].name == "search_tool"

    def test_multiple_tool_calls(self):
        tool1 = ToolCall(
            name="search_tool",
            description="Search",
            reasoning="Need to search",
            output={"results": []},
        )
        tool2 = ToolCall(
            name="weather_tool",
            description="Weather",
            reasoning="Need weather",
            output={"temp": "70F"},
        )

        turn = Turn(
            role="assistant",
            content="Using multiple tools...",
            tools_called=[tool1, tool2],
        )

        assert len(turn.tools_called) == 2
        assert turn.tools_called[0].name == "search_tool"
        assert turn.tools_called[1].name == "weather_tool"

    def test_empty_tools_called_list(self):
        turn = Turn(role="assistant", content="No tools used", tools_called=[])
        assert turn.tools_called == []


class TestTurnWithMetadata:

    def test_simple_metadata(self):
        metadata = {"model": "gpt-4", "tokens": 150}
        turn = Turn(
            role="assistant", content="Response", additional_metadata=metadata
        )
        assert turn.additional_metadata == metadata

    def test_complex_metadata(self):
        metadata = {
            "model": "gpt-4",
            "tokens": 150,
            "timestamp": "2024-01-01T10:00:00Z",
            "metrics": {"latency": 0.5, "confidence": 0.95},
            "tags": ["important", "customer-service"],
        }
        turn = Turn(
            role="assistant",
            content="Complex response",
            additional_metadata=metadata,
        )
        assert turn.additional_metadata == metadata
        assert turn.additional_metadata["metrics"]["confidence"] == 0.95

    def test_empty_metadata_dict(self):
        turn = Turn(role="user", content="Hello", additional_metadata={})
        assert turn.additional_metadata == {}

    def test_none_metadata(self):
        turn = Turn(role="user", content="Hello")
        assert turn.additional_metadata is None


class TestTurnWithUserID:

    def test_simple_user_id(self):
        turn = Turn(role="user", content="Hello", user_id="user123")
        assert turn.user_id == "user123"

    def test_uuid_user_id(self):
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        turn = Turn(role="user", content="Hello", user_id=user_id)
        assert turn.user_id == user_id

    def test_none_user_id(self):
        turn = Turn(role="user", content="Hello")
        assert turn.user_id is None


class TestTurnRepresentation:

    def test_repr_minimal(self):
        turn = Turn(role="user", content="Hello")
        repr_str = repr(turn)
        assert "role='user'" in repr_str
        assert "content='Hello'" in repr_str

    def test_repr_with_optional_fields(self):
        turn = Turn(
            role="assistant",
            content="Hi there!",
            user_id="user123",
            additional_metadata={"model": "gpt-4"},
        )
        repr_str = repr(turn)
        assert "role='assistant'" in repr_str
        assert "content='Hi there!'" in repr_str
        assert "user_id='user123'" in repr_str
        assert "additional_metadata=" in repr_str

    def test_repr_with_tools(self):
        tool_call = ToolCall(
            name="test_tool", description="Test", reasoning="Testing", output={}
        )
        turn = Turn(
            role="assistant", content="Using tool", tools_called=[tool_call]
        )
        repr_str = repr(turn)
        assert "tools_called=" in repr_str


class TestTurnEdgeCases:

    def test_very_long_content(self):
        long_content = "A" * 10000
        turn = Turn(role="user", content=long_content)
        assert len(turn.content) == 10000

    def test_special_characters_in_content(self):
        special_content = "Hello! 🌟 @#$%^&*() 你好 🎉"
        turn = Turn(role="user", content=special_content)
        assert turn.content == special_content

    def test_multiline_content(self):
        multiline_content = """This is a
        multiline message
        with different lines"""
        turn = Turn(role="user", content=multiline_content)
        assert "\n" in turn.content
        assert "multiline message" in turn.content

    def test_json_like_content(self):
        json_content = '{"key": "value", "number": 42}'
        turn = Turn(role="assistant", content=json_content)
        assert turn.content == json_content

    def test_code_content(self):
        code_content = """
def hello_world():
    print("Hello, World!")
    return "Hello"
        """
        turn = Turn(role="assistant", content=code_content)
        assert "def hello_world" in turn.content
        assert "print(" in turn.content


class TestTurnEquality:

    def test_identical_turns_are_equal(self):
        turn1 = Turn(role="user", content="Hello")
        turn2 = Turn(role="user", content="Hello")
        assert turn1.model_dump() == turn2.model_dump()

    def test_different_content_not_equal(self):
        turn1 = Turn(role="user", content="Hello")
        turn2 = Turn(role="user", content="Hi")
        assert turn1.model_dump() != turn2.model_dump()

    def test_different_roles_not_equal(self):
        turn1 = Turn(role="user", content="Hello")
        turn2 = Turn(role="assistant", content="Hello")
        assert turn1.model_dump() != turn2.model_dump()

    def test_different_metadata_not_equal(self):
        turn1 = Turn(role="user", content="Hello", additional_metadata={"a": 1})
        turn2 = Turn(role="user", content="Hello", additional_metadata={"a": 2})
        assert turn1.model_dump() != turn2.model_dump()


class TestTurnSerialization:

    def test_model_dump_basic(self):
        turn = Turn(role="user", content="Hello")
        dumped = turn.model_dump()

        assert dumped["role"] == "user"
        assert dumped["content"] == "Hello"
        assert dumped["user_id"] is None
        assert dumped["retrieval_context"] is None

    def test_model_dump_with_all_fields(self):
        tool_call = ToolCall(
            name="test_tool", description="Test", reasoning="Testing", output={}
        )

        turn = Turn(
            role="assistant",
            content="Response",
            user_id="user123",
            retrieval_context=["context1", "context2"],
            tools_called=[tool_call],
            additional_metadata={"key": "value"},
        )

        dumped = turn.model_dump()
        assert dumped["role"] == "assistant"
        assert dumped["content"] == "Response"
        assert dumped["user_id"] == "user123"
        assert len(dumped["retrieval_context"]) == 2
        assert len(dumped["tools_called"]) == 1
        assert dumped["additional_metadata"]["key"] == "value"

    def test_model_dump_exclude_none(self):
        turn = Turn(role="user", content="Hello")
        dumped = turn.model_dump(exclude_none=True)

        assert "user_id" not in dumped
        assert "retrieval_context" not in dumped
        assert "tools_called" not in dumped
        assert "additional_metadata" not in dumped


class TestTurnCamelCaseInitialization:

    def test_camelcase_field_initialization(self):
        # Test data variables
        role_value = "assistant"
        content_text = "Let me check the weather for you."
        user_id_value = "user123"
        retrieval_context_list = ["Weather data from API", "Current conditions"]
        metadata_dict = {
            "timestamp": "2024-01-01T10:00:00Z",
            "model": "gpt-4",
        }

        tool_call = ToolCall(
            name="weather_tool",
            description="Get weather info",
            reasoning="User asked about weather",
            output={"temperature": "75F", "condition": "sunny"},
            inputParameters={"location": "New York"},  # camelCase
        )

        turn = Turn(
            role=role_value,
            content=content_text,
            userId=user_id_value,  # camelCase
            retrievalContext=retrieval_context_list,  # camelCase
            toolsCalled=[tool_call],  # camelCase
            additionalMetadata=metadata_dict,  # camelCase
        )

        # Verify all fields are properly set using the same variables
        assert turn.role == role_value
        assert turn.content == content_text
        assert turn.user_id == user_id_value
        assert turn.retrieval_context == retrieval_context_list
        assert len(turn.tools_called) == 1
        assert turn.tools_called[0].name == "weather_tool"
        assert turn.additional_metadata == metadata_dict

    def test_mixed_case_initialization(self):
        # Test data variables
        role_value = "user"
        content_text = "Mixed case turn test"
        user_id_value = "mixedUser123"
        retrieval_context_list = ["Mixed context item"]
        metadata_dict = {"testMode": "mixed", "caseType": "both"}

        turn = Turn(
            role=role_value,
            content=content_text,
            userId=user_id_value,  # camelCase
            retrieval_context=retrieval_context_list,  # snake_case
            additionalMetadata=metadata_dict,  # camelCase
        )

        assert turn.role == role_value
        assert turn.content == content_text
        assert turn.user_id == user_id_value
        assert turn.retrieval_context == retrieval_context_list
        assert turn.additional_metadata == metadata_dict

    def test_turn_with_camelcase_tools(self):
        # Test data variables
        role_value = "assistant"
        content_text = "Using tools with camelCase parameters"

        camel_tool_name = "search_tool"
        camel_input_params = {
            "searchQuery": "camelCase search",
            "maxResults": 10,
        }
        camel_output = {"searchResults": ["result1", "result2"]}

        snake_tool_name = "calc_tool"
        snake_input_params = {"expression": "2 + 2", "precision": 2}
        snake_output = {"calculation_result": 4}

        # Test ToolCall with camelCase
        tool_call_camel = ToolCall(
            name=camel_tool_name,
            description="Search tool with camelCase",
            reasoning="Need to search",
            inputParameters=camel_input_params,  # camelCase
            output=camel_output,
        )

        # Test ToolCall with snake_case
        tool_call_snake = ToolCall(
            name=snake_tool_name,
            description="Calculator tool with snake_case",
            reasoning="Need to calculate",
            input_parameters=snake_input_params,  # snake_case
            output=snake_output,
        )

        turn = Turn(
            role=role_value,
            content=content_text,
            toolsCalled=[tool_call_camel, tool_call_snake],  # camelCase
        )

        assert turn.role == role_value
        assert turn.content == content_text
        assert len(turn.tools_called) == 2
        assert turn.tools_called[0].name == camel_tool_name
        assert turn.tools_called[0].input_parameters == camel_input_params
        assert turn.tools_called[1].name == snake_tool_name
        assert turn.tools_called[1].input_parameters == snake_input_params
