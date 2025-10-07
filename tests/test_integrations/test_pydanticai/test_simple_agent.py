import asyncio
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.prompt import Prompt
from deepeval.tracing.otel.test_exporter import test_exporter

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

confident_instrumentation_settings = ConfidentInstrumentationSettings(
    thread_id="test_thread_id_1",
    user_id="test_user_id_1",
    metadata={"streaming": True, "prompt_version": "00.00.11"},
    tags=["test_tag_1", "test_tag_2"],
    metric_collection="test_metric_collection_1",
    name="test_name_1",
    confident_prompt=prompt,
    trace_metric_collection="test_collection_1",

    is_test_mode=True,
)

agent = Agent(
    "openai:gpt-5",
    system_prompt="Be concise, reply with one sentence.",
    instrument=confident_instrumentation_settings,
    name="test_agent",
)

######################### GENERATE TEST JSON #################################

def generate_test_json():
    result = asyncio.run(agent.run("What are the LLMs?"))
    print(test_exporter.span_json_list)

generate_test_json()