from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from deepeval.integrations.agentcore import instrument_agentcore


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "tokyo": "Sunny, 72F",
        "london": "Rainy, 55F",
        "paris": "Cloudy, 62F",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}"
    )


@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    time_data = {
        "tokyo": "3:00 PM JST",
        "london": "7:00 AM GMT",
        "paris": "8:00 AM CET",
    }
    return time_data.get(city.lower(), f"Time data not available for {city}")


def init_multiple_tools_agentcore(
    name: str = "agentcore-multiple-tools-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    instrument_agentcore(
        name=name,
        tags=tags or ["agentcore", "multiple-tools"],
        metadata=metadata or {"test_type": "multiple_tools"},
        thread_id=thread_id,
        user_id=user_id,
        is_test_mode=True,
    )

    app = BedrockAgentCoreApp()
    agent = Agent(model="amazon.nova-lite-v1:0", tools=[get_weather, get_time])

    @app.entrypoint
    def invoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = (
            "You have access to weather and time tools. "
            "When asked about weather, use get_weather. "
            "When asked about time, use get_time. Be concise. "
        )
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = (
            "You have access to weather and time tools. "
            "When asked about weather, use get_weather. "
            "When asked about time, use get_time. Be concise. "
        )
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_multiple_tools_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_multiple_tools_agentcore()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_multiple_tools_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_multiple_tools_agentcore()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
