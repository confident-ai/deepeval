import os
import tempfile
from deepeval.tracing.utils import run_in_mode, compare_trace_files
from langgraph_app import execute_agent

def test_exec_agent_logs():
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_path = tmp.name
        tmp.close()
        try:
                run_in_mode("gen", execute_agent, file_path=tmp_path)
                expected_path = os.path.join(os.path.dirname(__file__), "langgraph_app.json")
                compare_trace_files(expected_path, tmp_path)
        finally:
                if os.path.exists(tmp_path):
                        os.remove(tmp_path)