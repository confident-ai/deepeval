import asyncio
from deepeval.tracing import observe


@observe()
def level_1():

    @observe()
    def level_2():

        @observe(type="tool", description="tool call description")
        def tool_call(input: str):
            return "tool call response"

        tool_call("test")
        return "Level 2 response"

    level_2()

    return "Level 1 response"


from deepeval.tracing.trace_test_manager import trace_testing_manager


async def test_tools_called_propogation():
    try:
        trace_testing_manager.test_name = "test_tools_called_propogation"
        level_1()
        test_dict = await trace_testing_manager.wait_for_test_dict()

        assert len(test_dict["baseSpans"][1]["toolsCalled"]) > 0
        assert test_dict["baseSpans"][0].get("toolsCalled") is None

    finally:
        trace_testing_manager.test_dict = None
        trace_testing_manager.test_name = None
