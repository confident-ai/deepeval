from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

metric = TaskCompletionMetric(
    threshold=0.7, model="gpt-4o", include_reason=True
)
test_case = LLMTestCase(
    input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
    actual_output=(
        "Day 1: Eiffel Tower, dinner at Le Jules Verne. "
        "Day 2: Louvre Museum, lunch at Angelina Paris. "
        "Day 3: Montmartre, evening at a wine bar."
    ),
    tools_called=[
        ToolCall(
            name="Itinerary Generator",
            description="Creates travel plans based on destination and duration.",
            input_parameters={"destination": "Paris", "days": 3},
            output=[
                "Day 1: Eiffel Tower, Le Jules Verne.",
                "Day 2: Louvre Museum, Angelina Paris.",
                "Day 3: Montmartre, wine bar.",
            ],
        ),
        ToolCall(
            name="Restaurant Finder",
            description="Finds top restaurants in a city.",
            input_parameters={"city": "Paris"},
            output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
        ),
    ],
)

# metric.measure(test_case)
# print(metric.score)
# print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])
