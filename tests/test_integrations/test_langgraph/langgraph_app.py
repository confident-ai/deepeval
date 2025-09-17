from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler, tool

# from deepeval.tracing.tracing import observe
from langchain_openai import ChatOpenAI
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")


@tool(metric_collection="test_collection_1")
def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric_collection": "test_collection_1", "prompt": prompt},
)

agent = create_react_agent(
    model=llm,
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
        config={
            "callbacks": [
                CallbackHandler(
                    name="langgraph-test",
                    tags=["langgraph", "test"],
                    metadata={"environment": "test"},
                    thread_id="123",
                    user_id="456",
                    metric_collection="task_completion",
                )
            ],
        },
    )
