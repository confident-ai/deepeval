from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.test_run.cache import CachedTestCase


def test_cache_key_differs_for_different_tools_called():
    tc1 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
        tools_called=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
        expected_tools=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
    )

    tc2 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
        tools_called=[
            ToolCall(name="search_web", input_parameters={}, output=None),
            ToolCall(name="get_weather", input_parameters={}, output=None),
        ],
        expected_tools=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
    )

    key1 = CachedTestCase.create_cache_key(tc1)
    key2 = CachedTestCase.create_cache_key(tc2)

    assert key1 != key2


def test_cache_key_differs_for_different_trace_dict():
    tc1 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
    )
    tc1._trace_dict = {"spans": [{"name": "llm_call", "output": "sunny"}]}

    tc2 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
    )
    tc2._trace_dict = {
        "spans": [{"name": "tool_call", "output": "weather_api_result"}]
    }

    key1 = CachedTestCase.create_cache_key(tc1)
    key2 = CachedTestCase.create_cache_key(tc2)

    assert key1 != key2


def test_cache_key_matches_for_same_execution_state():
    tc1 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
        tools_called=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
        expected_tools=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
    )
    tc1._trace_dict = {"spans": [{"name": "llm_call", "output": "sunny"}]}

    tc2 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
        tools_called=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
        expected_tools=[
            ToolCall(name="get_weather", input_parameters={}, output=None)
        ],
    )
    tc2._trace_dict = {"spans": [{"name": "llm_call", "output": "sunny"}]}

    key1 = CachedTestCase.create_cache_key(tc1)
    key2 = CachedTestCase.create_cache_key(tc2)

    assert key1 == key2


def test_cache_key_matches_for_text_only_case():
    tc1 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
    )
    tc2 = LLMTestCase(
        input="What is the weather?",
        actual_output="It's sunny",
    )

    key1 = CachedTestCase.create_cache_key(tc1)
    key2 = CachedTestCase.create_cache_key(tc2)

    assert key1 == key2
