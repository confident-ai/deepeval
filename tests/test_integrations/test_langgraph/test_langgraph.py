from deepeval.tracing.utils import run_in_test_mode
from langgraph_app import execute_agent

def test_exec_agent_logs(capsys):
        run_in_test_mode(execute_agent, file_path="langgraph_app.json")
        out, err = capsys.readouterr()
        try:
            assert "Trace body does not match expected file:" not in out
        except AssertionError:
            print(out)
            raise