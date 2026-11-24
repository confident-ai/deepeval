import pytest

from deepteam.code_scanner import CodeScanner, collect_files

# Real-LLM smoke test. Excluded by default (pyproject: addopts = -m 'not
# skip_test'). Run explicitly with: pytest -m skip_test  (needs OPENAI_API_KEY).


@pytest.mark.skip_test
def test_real_scan_flags_eval(tmp_path):
    (tmp_path / "agent.py").write_text(
        "def run(resp):\n    return eval(resp)\n"
    )
    findings = CodeScanner(model="gpt-4o").scan(collect_files(str(tmp_path)))
    assert any(f.vulnerability == "Unexpected Code Execution" for f in findings)
