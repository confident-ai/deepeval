import os
import time
import json
import asyncio
from pathlib import Path
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.prompt import Prompt
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_context import trace
from deepeval.tracing.otel.utils import to_hex_string
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
)

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

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

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "simple_agent.json")


def generate_test_json():
    try:
        result = asyncio.run(agent.run("What are the LLMs?"))
        time.sleep(7)
        with open(json_path, "w") as f:
            json.dump(test_exporter.get_span_json_list(), f, indent=2)
    finally:
        test_exporter.clear_span_json_list()


# generate_test_json()


def test_simple_agent():
    with trace() as current_trace:
        asyncio.run(agent.run("What are the LLMs?"))
        time.sleep(7)
        try:
            expected_dict = load_trace_data(json_path)
            actual_dict = test_exporter.get_span_json_list()
            assert assert_json_object_structure(expected_dict, actual_dict)
            assert (
                current_trace.uuid == actual_dict[0]["context"]["trace_id"][2:]
            )
        finally:
            test_exporter.clear_span_json_list()
