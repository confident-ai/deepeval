from tests.test_integrations.utils import generate_test_json
from langgraph_app import execute_agent

generate_test_json(execute_agent, "langgraph_app.json")

# execute_agent()
