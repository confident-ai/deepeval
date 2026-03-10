from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from deepeval.integrations.agentcore import instrument_agentcore


def init_simple_agentcore(
    name: str = "agentcore-simple-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    instrument_agentcore(
        name=name,
        tags=tags or ["agentcore", "simple"],
        metadata=metadata or {"test_type": "simple"},
        thread_id=thread_id,
        user_id=user_id,
        is_test_mode=True,
    )

    app = BedrockAgentCoreApp()
    agent = Agent(model="amazon.nova-lite-v1:0")

    @app.entrypoint
    def invoke(payload: dict):
        user_message = payload.get("prompt", "Hello!")
        instruction = "Be concise, reply with one short sentence only. "
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "Hello!")
        instruction = "Be concise, reply with one short sentence only. "
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_simple_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_simple_agentcore()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_simple_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_simple_agentcore()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
