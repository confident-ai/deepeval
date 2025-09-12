from tests.test_integrations.utils import generate_test_json
from langchain_app import execute_agent

generate_test_json(execute_agent, "langchain_app.json")
