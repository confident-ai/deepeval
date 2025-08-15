from deepeval.metrics import TaskCompletionMetric

task_completion_metric = TaskCompletionMetric()

from deepeval.tracing import observe
from deepeval.dataset import EvaluationDataset, Golden


@observe()
def your_ai_agent_tool():
    return "tool call result"


# Supply task completion
@observe(metrics=[task_completion_metric])
def your_ai_agent(input):
    tool_call_result = your_ai_agent_tool()
    return "Tool Call Result: " + tool_call_result


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

# Loop through dataset
for golden in dataset.evals_iterator():
    your_ai_agent(golden.input)
