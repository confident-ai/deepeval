import time
from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.tracing import Observer


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
            "messages": [
                {"role": "user", "content": "what is the weather in sf"}
            ]
        },
        config={"callbacks": [CallbackHandler()]},
    )


def api_call(is_failure: bool):

    if is_failure:
        raise Exception("Test error")

    return "Success"


def app(is_failure: bool):
    with Observer(
        span_type="custom",
        func_name="api_call",
    ) as observer:
        try:
            api_call(is_failure=is_failure)
        except Exception as e:
            raise e
        execute_agent()


app(is_failure=False)
