from tests.test_integrations.utils import generate_test_json
from run_main import execute_all

generate_test_json(execute_all, "agents_app.json")

# open ai agents testing heirarchy
# 1. run
# 2. run_sync
# 3. run_stream
# 4. Agent
# 5. tool

# - all the trace attributes inside them run methods
# - all the agent attributes
# - all the tool attributes

# - Agents trace context
#     - it should have one multi agent too, with all the attributes 
#     - overriting of the trace attributes
