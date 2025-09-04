from agents import Agent, Runner, add_trace_processor, RunConfig
from deepeval.openai_agents import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())

# Replace with your agent code
agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(
    starting_agent=agent, 
    input="Write a haiku about recursion in programming.", 
    run_config=RunConfig(trace_metadata={"metric_collection": "test_collection_1"})
)
