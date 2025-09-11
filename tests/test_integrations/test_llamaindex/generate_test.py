from llamaindex_app import execute_agent
from tests.test_integrations.utils import generate_test_json

generate_test_json(execute_agent, "llamaindex_app.json")
