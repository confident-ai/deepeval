import os

from strands import Agent
from strands.models.openai import OpenAIModel

from deepeval.integrations.strands import instrument_strands


_DEFAULT_MODEL_ID = os.environ.get("STRANDS_TEST_MODEL", "gpt-4o-mini")


def _build_openai_model() -> OpenAIModel:
    """Strands' OpenAI provider. Read the API key lazily so tests that
    don't actually invoke the model (skipped via pytest markers) don't
    need ``OPENAI_API_KEY`` set just to import the app module."""
    return OpenAIModel(
        client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
        model_id=_DEFAULT_MODEL_ID,
        params={"temperature": 0.0},
    )


def init_simple_strands(
    name: str = "strands-simple-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Wire the deepeval OTel pipeline and build a Strands agent.

    All kwargs are trace-level. Span-level configuration belongs at the
    call site via ``with next_*_span(...)`` blocks or
    ``update_current_span(...)`` from inside a Strands ``@tool`` body.
    """
    instrument_strands(
        name=name,
        tags=tags or ["strands", "simple"],
        metadata=metadata or {"test_type": "simple"},
        thread_id=thread_id,
        user_id=user_id,
    )

    agent = Agent(model=_build_openai_model())

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
        invoke_func = init_simple_strands()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_simple_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_simple_strands()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
