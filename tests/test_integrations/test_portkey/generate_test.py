from portkey_app import execute_chat_completion
from tests.test_integrations.utils import generate_test_json

generate_test_json(execute_chat_completion, "portkey_app.json")