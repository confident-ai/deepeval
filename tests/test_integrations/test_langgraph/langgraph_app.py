from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.utils import run_in_test_mode

def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

def execute_agent():
    return agent.invoke(
        input={
            "messages": [{"role": "user", "content": "what is the weather in sf"}]
        },
        config={
            "callbacks": [CallbackHandler(metric_collection="task_completion")]
        },
    )

run_in_test_mode(func=execute_agent, file_path="langgraph_app.json")