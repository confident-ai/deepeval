from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig

test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            description="Tool to search for information on the web.",
            input_parameters={
                "search_query": "Trump first raised tariffs year"
            },
        ),
        ToolCall(
            name="History FunFact Tool",
            description="Tool to provide a fun fact about the topic.",
            input_parameters={"topic": "Trump tariffs"},
        ),
    ],
)

no_description_test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            input_parameters={
                "search_query": "Trump first raised tariffs year"
            },
        ),
        ToolCall(
            name="History FunFact Tool",
            input_parameters={"topic": "Trump tariffs"},
        ),
    ],
)

no_input_test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            description="Tool to search for information on the web.",
        ),
        ToolCall(
            name="History FunFact Tool",
            description="Tool to provide a fun fact about the topic.",
        ),
    ],
)

empty_test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[],
)

evaluate(
    test_cases=[
        test_case,
        empty_test_case,
        no_description_test_case,
        no_input_test_case,
    ],
    metrics=[ArgumentCorrectnessMetric(threshold=0.4, model="gpt-4o")],
    async_config=AsyncConfig(run_async=True, max_concurrent=2),
)

metric = ArgumentCorrectnessMetric(threshold=0.4)
for test_case in [
    test_case,
    no_input_test_case,
    empty_test_case,
    no_description_test_case,
]:
    metric.measure(test_case)
    print(metric.score, metric.reason)
