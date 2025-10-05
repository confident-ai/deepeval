# this is not working
from crewai import Task, Crew

from deepeval.integrations.crewai import Agent
from deepeval.integrations.crewai import instrument_crewai
from deepeval.metrics import AnswerRelevancyMetric
import os
import json
import pytest
from tests.test_integrations.utils import assert_json_object_structure, load_trace_data
from tests.test_integrations.manager import trace_testing_manager
import asyncio

# instrument_crewai()

answer_relavancy_metric = AnswerRelevancyMetric()

agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metric_collection="test_collection_1",
)

task = Task(
    description="Explain the given topic",
    expected_output="A clear and concise explanation.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

result = crew.kickoff({"input": "What are the LLMs?"})


def execute_agent():
    result = crew.kickoff({"input": "What are the LLMs?"})
    return result

################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, 'crewai.json')

@pytest.mark.skip(reason="CrewAI integration is deprecated.")
async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    try:
        trace_testing_manager.test_name = json_path
        execute_agent()
        actual_dict = await trace_testing_manager.wait_for_test_dict()
        expected_dict = load_trace_data(json_path)

        print(actual_dict)
        
        assert assert_json_object_structure(expected_dict, actual_dict)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None

################################ Generate Actual JSON Dump Code #################################

async def generate_actual_json_dump():
    try:
        trace_testing_manager.test_name = json_path
        execute_agent()
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        with open(json_path, 'w') as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None

# asyncio.run(generate_actual_json_dump())
