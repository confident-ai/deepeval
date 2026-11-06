import math

import pytest


def record(identifier="a", **changes):
    return {
        "id": identifier,
        "parent_id": None,
        "kind": "tool",
        "name": "verify",
        "status": "success",
        "start": 1.0,
        "end": 2.0,
        "payload": {"input": {"account": "alice"}},
        **changes,
    }


def snapshot(records, **kwargs):
    from deepeval.metrics.community.execution_contract.evidence import Snapshot

    return Snapshot.from_records(records, complete=True, **kwargs)


def test_snapshot_detaches_nested_input_and_returned_values():
    source = record()
    result = snapshot([source])
    source["payload"]["input"]["account"] = "bob"
    value = result.spans[0].payload
    value["input"]["account"] = "mallory"
    assert result.spans[0].payload["input"]["account"] == "alice"
    with pytest.raises((AttributeError, TypeError)):
        result.spans[0].name = "pay"


@pytest.mark.parametrize(
    "records",
    [
        [record(), record()],
        [record(parent_id="a")],
        [record("a", parent_id="b"), record("b", parent_id="a")],
        [record(parent_id="missing")],
        [record(start=math.nan)],
        [record(end=math.inf)],
        [record(start=3, end=2)],
        [record(start=True)],
        [record(payload={"nonfinite": math.nan})],
        [record(payload={1: "not a JSON object key"})],
    ],
)
def test_invalid_evidence_is_rejected(records):
    with pytest.raises(ValueError):
        snapshot(records)


def test_explicit_subtree_boundary_allows_external_parent():
    result = snapshot(
        [record(parent_id="outside")], external_parents=("outside",)
    )
    assert result.spans[0].parent_id == "outside"
    assert result.external_parents == ("outside",)


def test_terminal_claim_cannot_hide_in_progress_span():
    result = snapshot([record(status="in_progress", end=None)])
    assert not result.complete
    assert result.incomplete_reasons == ("unfinished_span:a",)


def test_sibling_order_does_not_change_snapshot_fingerprint():
    a, b = record("a"), record("b")
    assert snapshot([a, b]).fingerprint == snapshot([b, a]).fingerprint
    assert (
        snapshot([a]).fingerprint
        != snapshot([record(payload={"x": 0})]).fingerprint
    )


def test_declared_completeness_changes_fingerprint():
    from deepeval.metrics.community.execution_contract.evidence import Snapshot

    assert (
        snapshot([record()]).fingerprint
        != Snapshot.from_records([record()], complete=False).fingerprint
    )


def test_evaluation_keeps_partial_snapshot_fingerprint():
    from deepeval.metrics.community.execution_contract import (
        Contract,
        Occurrence,
        Selector,
        evaluate,
    )

    evidence = snapshot([record(status="in_progress", end=None)])
    report = evaluate(
        evidence, Contract((Occurrence("verify", Selector(name="verify")),))
    )
    assert report.evidence_fingerprint == evidence.fingerprint


def test_public_evaluation_rejects_forged_completeness():
    from dataclasses import replace
    from deepeval.metrics.community.execution_contract import (
        Contract,
        Occurrence,
        Selector,
        evaluate,
    )

    evidence = snapshot([record(status="in_progress", end=None)])
    with pytest.raises(ValueError, match="derived"):
        evaluate(
            replace(evidence, complete=True),
            Contract((Occurrence("verify", Selector()),)),
        )


def test_limits_reject_instead_of_truncating():
    from deepeval.metrics.community.execution_contract.evidence import (
        EvidenceLimits,
    )

    with pytest.raises(ValueError, match="span limit"):
        snapshot([record("a"), record("b")], limits=EvidenceLimits(max_spans=1))
    with pytest.raises(ValueError, match="payload"):
        snapshot([record()], limits=EvidenceLimits(max_payload_bytes=2))
    with pytest.raises(ValueError, match="depth"):
        snapshot(
            [record("a"), record("b", parent_id="a")],
            limits=EvidenceLimits(max_depth=1),
        )


def test_declared_clock_domain_is_preserved():
    assert (
        snapshot([record()], clock_domain="process:abc").clock_domain
        == "process:abc"
    )
