from pathlib import Path
import runpy

from deepeval.metrics.community.execution_contract import evaluate
from deepeval.metrics.community.execution_contract.artifacts import (
    read_artifact,
)


def test_native_checkout_example_replays_identical_reports(tmp_path):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/community/execution_contract/evaluate.py"
    )
    namespace = runpy.run_path(str(path), run_name="execution_contract_example")
    reports = namespace["run_example"](tmp_path / "replay")
    assert {key: report.status.value for key, report in reports.items()} == {
        "valid": "pass",
        "wrong_subject": "fail",
        "recovered": "pass",
        "unrecovered": "fail",
    }
    contract = read_artifact(tmp_path / "replay/contract.json")
    for scenario, report in reports.items():
        snapshot = read_artifact(tmp_path / f"replay/{scenario}.evidence.json")
        assert evaluate(snapshot, contract) == report
