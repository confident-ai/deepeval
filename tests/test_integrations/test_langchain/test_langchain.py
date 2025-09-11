import sys
import os
import tempfile
import time
from tests.test_integrations.utils import compare_trace_files
from langchain_app import execute_agent
from deepeval.tracing import trace_manager

def test_exec_agent_logs():
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_path = tmp.name
        tmp.close()
        
        try:
            original_argv = list(sys.argv)
            sys.argv = ["--mode=gen", f"--file-name={tmp_path}"]
            execute_agent()
            time.sleep(10)
            trace_manager._process_trace_queue()
            sys.argv = original_argv
            expected_path = os.path.join(os.path.dirname(__file__), "langchain_app.json")
            compare_trace_files(expected_path, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"Removed temp file: {tmp_path}")