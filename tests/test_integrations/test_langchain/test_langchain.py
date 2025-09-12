import sys
import os
import tempfile
import time
from tests.test_integrations.utils import compare_trace_files
from langchain_app import execute_agent


def test_exec_agent_logs():

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = tmp.name
    tmp.close()

    try:
        original_argv = list(sys.argv)
        sys.argv = [
            "--deepeval-trace-mode=gen",
            f"--deepeval-trace-file-name={tmp_path}",
        ]
        execute_agent()
        sys.argv = original_argv
        expected_path = os.path.join(
            os.path.dirname(__file__), "langchain_app.json"
        )
        compare_trace_files(expected_path, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
