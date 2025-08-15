from agents import Agent, Runner, add_trace_processor
from deepeval.openai_agents import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())

# Replace with your agent code
agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
