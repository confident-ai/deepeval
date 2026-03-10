from typing import List, Dict
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from deepeval.integrations.agentcore import instrument_agentcore
from deepeval.prompt import Prompt
from deepeval.metrics import BaseMetric


@tool
def special_tool(query: str) -> str:
    """A tool to test tool metric collections."""
    return f"Processed: {query}"


def init_evals_agentcore(
    name: str = "agentcore-evals-test",
    tags: List[str] = None,
    metadata: Dict = None,
    thread_id: str = None,
    user_id: str = None,
    metric_collection: str = None,
    agent_metric_collection: str = None,
    llm_metric_collection: str = None,
    tool_metric_collection_map: Dict = None,
    trace_metric_collection: str = None,
    confident_prompt: Prompt = None,
    agent_metrics: List[BaseMetric] = None,
):
    instrument_agentcore(
        name=name,
        tags=tags or ["agentcore", "evals"],
        metadata=metadata or {"test_type": "evals"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
        agent_metric_collection=agent_metric_collection,
        llm_metric_collection=llm_metric_collection,
        tool_metric_collection_map=tool_metric_collection_map,
        trace_metric_collection=trace_metric_collection,
        confident_prompt=confident_prompt,
        agent_metrics=agent_metrics,
        is_test_mode=True,
    )

    app = BedrockAgentCoreApp()
    agent = Agent(model="amazon.nova-lite-v1:0", tools=[special_tool])

    @app.entrypoint
    def invoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = "You are a helpful assistant. Be concise. "
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = "You are a helpful assistant. Be concise. "
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_agentcore()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_agentcore()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
