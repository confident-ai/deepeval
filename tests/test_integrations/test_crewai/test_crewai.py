import sys
import os
import tempfile
import time
from tests.test_integrations.utils import compare_trace_files
from crewai_app import execute_agent

def test_exec_agent_logs():
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_path = tmp.name
        tmp.close()
        
        try:
            original_argv = list(sys.argv)
            sys.argv = ["--mode=gen", f"--file-name={tmp_path}"]
            execute_agent()
            time.sleep(10)
            sys.argv = original_argv
            expected_path = os.path.join(os.path.dirname(__file__), "crewai_app.json")
            compare_trace_files(expected_path, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"Removed temp file: {tmp_path}")