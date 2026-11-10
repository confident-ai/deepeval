import json
import os
import subprocess
import sys

import pytest

from deepeval.metrics.community.execution_contract import (
    Contract,
    Occurrence,
    Selector,
    Snapshot,
    evaluate,
)
from deepeval.metrics.community.execution_contract.artifacts import (
    write_artifact,
)


def report(name="verify", *, complete=True, required=True):
    snapshot = Snapshot.from_records(
        [
            {
                "id": "one",
                "kind": "tool",
                "name": name,
                "status": "success",
                "start": 1,
                "end": 2,
            },
        ],
        complete=complete,
    )
    contract = Contract(
        (
            Occurrence(
                "verify", Selector(name="verify"), maximum=1, required=required
            ),
        )
    )
    return evaluate(snapshot, contract)


def test_compare_distinguishes_failure_from_missing_evidence():
    from deepeval.metrics.community.execution_contract.comparison import (
        compare_reports,
    )

    passing = report()
    failed = report("pay")
    unknown = report(complete=False)
    assert (
        compare_reports(passing, failed).transitions[0].change == "new_failure"
    )
    assert (
        compare_reports(failed, passing).transitions[0].change
        == "resolved_failure"
    )
    assert (
        compare_reports(passing, unknown).transitions[0].change
        == "lost_evidence"
    )
    assert (
        compare_reports(unknown, passing).transitions[0].change
        == "resolved_unknown"
    )
    assert compare_reports(passing, unknown).regressed
    assert not compare_reports(failed, passing).regressed


def test_compare_rejects_different_contracts_and_tracks_optional_diagnostics():
    from deepeval.metrics.community.execution_contract.comparison import (
        compare_reports,
    )

    with pytest.raises(ValueError, match="contract"):
        compare_reports(report(), report(required=False))
    optional = compare_reports(
        report(required=False), report("pay", required=False)
    )
    assert optional.transitions[0].change == "new_failure"
    assert not optional.regressed


def run_cli(*args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "deepeval.metrics.community.execution_contract",
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "DEEPEVAL_TELEMETRY_OPT_OUT": "1"},
    )


@pytest.mark.parametrize(
    "name,complete,code",
    [("verify", True, 0), ("pay", True, 1), ("verify", False, 2)],
)
def test_fresh_process_replay_and_exit_codes(tmp_path, name, complete, code):
    contract = Contract(
        (Occurrence("verify", Selector(name="verify"), maximum=1),)
    )
    snapshot = Snapshot.from_records(
        [
            {
                "id": "one",
                "kind": "tool",
                "name": name,
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {"secret": "do-not-print"},
            },
        ],
        complete=complete,
    )
    contract_path, evidence_path, result_path = [
        tmp_path / file
        for file in ("contract.json", "evidence.json", "result.json")
    ]
    write_artifact(contract_path, contract)
    write_artifact(evidence_path, snapshot)
    result = run_cli(
        "replay",
        "--contract",
        contract_path,
        "--evidence",
        evidence_path,
        "--output",
        result_path,
    )
    assert result.returncode == code, result.stderr
    assert json.loads(result.stdout) == evaluate(snapshot, contract).to_dict()
    assert "do-not-print" not in result.stdout + result.stderr
    from deepeval.metrics.community.execution_contract.artifacts import (
        read_artifact,
    )

    assert read_artifact(result_path) == evaluate(snapshot, contract)


def test_cli_compare_reports_fails_on_regression(tmp_path):
    baseline, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    write_artifact(baseline, report())
    write_artifact(candidate, report("pay"))
    result = run_cli(
        "compare", "--baseline", baseline, "--candidate", candidate
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["regressed"] is True


def test_cli_rejects_invalid_artifacts_without_traceback_or_payload_echo(
    tmp_path,
):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"secret":"do-not-print", invalid}')
    result = run_cli("replay", "--contract", invalid, "--evidence", invalid)
    assert result.returncode == 3
    assert result.stdout == ""
    assert "do-not-print" not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_cannot_replace_evidence_with_report(tmp_path):
    contract_path, evidence_path = (
        tmp_path / "contract.json",
        tmp_path / "evidence.json",
    )
    contract = Contract((Occurrence("verify", Selector()),))
    snapshot = Snapshot.from_records([], complete=True)
    write_artifact(contract_path, contract)
    write_artifact(evidence_path, snapshot)
    before = evidence_path.read_bytes()
    result = run_cli(
        "replay",
        "--contract",
        contract_path,
        "--evidence",
        evidence_path,
        "--output",
        evidence_path,
    )
    assert result.returncode == 3
    assert evidence_path.read_bytes() == before
