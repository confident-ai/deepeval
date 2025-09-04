from agents import Agent, add_trace_processor
from deepeval.openai_agents import DeepEvalTracingProcessor, Runner

add_trace_processor(DeepEvalTracingProcessor())

# Replace with your agent code
agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(
    starting_agent=agent,
    input="Write a haiku about recursion in programming.",
    metric_collection="task_completion",
)
