import json

import pytest

from deepeval.metrics.community.execution_contract import (
    Budget,
    Contract,
    EvaluationLimits,
    Occurrence,
    Predicate,
    Prerequisite,
    Recovery,
    Selector,
    Snapshot,
    evaluate,
)


def sample_contract():
    subject = (("/input/account", "/input/account"),)
    return Contract(
        (
            Occurrence("pay", Selector(name="pay"), minimum=1, maximum=2),
            Prerequisite(
                "authorization",
                Selector(name="pay"),
                Selector(
                    name="verify",
                    status="success",
                    predicates=(Predicate("/output/valid", "eq", True),),
                ),
                bindings=subject,
                max_age=60,
                invalidated_by=Selector(name="revoke"),
            ),
            Recovery(
                "refund",
                Selector(name="pay", status="error"),
                Selector(name="refund", status="success"),
                bindings=subject,
                within=30,
                before=Selector(name="notify"),
                required=False,
            ),
            Budget(
                "tokens",
                Selector(kind="llm"),
                "/resources/total_tokens",
                maximum=300,
                exclusive=True,
                required=False,
            ),
        ),
        limits=EvaluationLimits(max_work=10000),
    )


def evidence(*, partial=False):
    return Snapshot.from_records(
        [
            {
                "id": "verify",
                "kind": "tool",
                "name": "verify",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {
                    "input": {"account": "alice"},
                    "output": {"valid": True},
                },
            },
            {
                "id": "payment",
                "kind": "tool",
                "name": "pay",
                "status": "in_progress" if partial else "success",
                "start": 3,
                "end": None if partial else 4,
                "payload": {
                    "input": {"account": "alice"},
                    "sensitive": "user secret",
                },
            },
        ],
        complete=True,
        clock_domain="test-clock",
    )


def test_contract_roundtrip_keeps_all_rules_and_limits():
    from deepeval.metrics.community.execution_contract.serialization import (
        contract_to_data,
        contract_from_data,
        contract_fingerprint,
    )

    original = sample_contract()
    reconstructed = contract_from_data(contract_to_data(original))
    assert reconstructed == original
    assert contract_fingerprint(reconstructed) == contract_fingerprint(original)
    changed = Contract((Occurrence("pay", Selector(name="other")),))
    assert contract_fingerprint(changed) != contract_fingerprint(original)


@pytest.mark.parametrize("partial", [False, True])
def test_evidence_artifact_roundtrip_preserves_report_and_provenance(
    tmp_path, partial
):
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )

    snapshot = evidence(partial=partial)
    path = tmp_path / "evidence.json"
    write_artifact(path, snapshot)
    recovered = read_artifact(path, expected_kind="snapshot")
    assert recovered == snapshot
    assert recovered.fingerprint == snapshot.fingerprint
    assert evaluate(recovered, sample_contract()) == evaluate(
        snapshot, sample_contract()
    )


def test_report_roundtrip_omits_raw_payload(tmp_path):
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )

    report = evaluate(evidence(), sample_contract())
    path = tmp_path / "report.json"
    write_artifact(path, report)
    assert read_artifact(path, expected_kind="report") == report
    assert "user secret" not in path.read_text()
    assert "alice" not in path.read_text()


def test_report_is_bound_to_contract_as_well_as_evidence():
    first = evaluate(evidence(), sample_contract())
    second = evaluate(
        evidence(), Contract((Occurrence("pay", Selector(name="other")),))
    )
    assert first.evidence_fingerprint == second.evidence_fingerprint
    assert first.contract_fingerprint != second.contract_fingerprint


def test_report_explains_capture_scope_without_including_payloads():
    from deepeval.metrics.community.execution_contract.artifacts import (
        encode_artifact,
        decode_artifact,
    )

    snapshot = Snapshot.from_records(
        [
            {
                "id": "one",
                "parent_id": "outside",
                "kind": "tool",
                "name": "pay",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {"secret": "private"},
            },
        ],
        complete=False,
        external_parents=("outside",),
    )
    report = evaluate(snapshot, sample_contract())
    assert report.to_dict()["capture"] == {
        "span_count": 1,
        "complete": False,
        "external_parent_count": 1,
        "clock_domain_known": False,
    }
    assert "private" not in json.dumps(report.to_dict())
    assert decode_artifact(encode_artifact(report)) == report


def test_corruption_is_rejected_before_semantic_loading(tmp_path):
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )

    path = tmp_path / "evidence.json"
    write_artifact(path, evidence())
    raw = json.loads(path.read_text())
    raw["data"]["spans"][0]["name"] = "tampered"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        read_artifact(path)


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', "[" * 1500 + "0" + "]" * 1500]
)
def test_malformed_json_is_rejected_without_echoing_content(tmp_path, text):
    from deepeval.metrics.community.execution_contract.artifacts import (
        read_artifact,
    )

    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_artifact(path)


def test_version_kind_and_file_size_are_checked(tmp_path):
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
        ArtifactLimits,
    )

    path = tmp_path / "contract.json"
    write_artifact(path, sample_contract())
    with pytest.raises(ValueError, match="kind"):
        read_artifact(path, expected_kind="snapshot")
    with pytest.raises(ValueError, match="byte limit"):
        read_artifact(path, limits=ArtifactLimits(max_bytes=50))
    raw = json.loads(path.read_text())
    raw["version"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="version"):
        read_artifact(path)


def test_unknown_contract_fields_and_invalid_predicates_are_rejected():
    from deepeval.metrics.community.execution_contract.serialization import (
        contract_from_data,
        contract_to_data,
    )

    raw = contract_to_data(sample_contract())
    raw["rules"][0]["maximun"] = 10
    with pytest.raises(ValueError, match="fields"):
        contract_from_data(raw)
    raw = contract_to_data(sample_contract())
    raw["rules"][1]["prerequisite"]["predicates"][0]["operator"] = "python"
    with pytest.raises(ValueError, match="operator"):
        contract_from_data(raw)


def test_failed_export_keeps_existing_file_and_cleans_temporary_files(
    tmp_path, monkeypatch
):
    from deepeval.metrics.community.execution_contract import artifacts

    path = tmp_path / "contract.json"
    path.write_text("old")
    before = {item.name for item in tmp_path.iterdir()}

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError):
        artifacts.write_artifact(path, sample_contract())
    assert path.read_text() == "old"
    assert {item.name for item in tmp_path.iterdir()} == before


@pytest.mark.parametrize("pointer", ["", "/" + "x" * 2000])
def test_report_replay_preserves_valid_missing_json_pointers(pointer):
    from deepeval.metrics.community.execution_contract.artifacts import (
        encode_artifact,
        decode_artifact,
    )

    snapshot = Snapshot.from_records(
        [
            {
                "id": "one",
                "kind": "tool",
                "name": "call",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": None,
            },
        ],
        complete=True,
    )
    report = evaluate(
        snapshot, Contract((Budget("budget", Selector(), pointer, 1),))
    )
    assert report.status.value == "unknown"
    assert decode_artifact(encode_artifact(report)) == report
