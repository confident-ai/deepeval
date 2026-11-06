from itertools import repeat

import pytest

from deepeval.metrics.community.execution_contract import (
    Budget,
    EvidenceLimits,
    Predicate,
    Prerequisite,
    Selector,
    Snapshot,
)


def test_repeated_boundary_generator_is_bounded_even_without_unique_growth():
    with pytest.raises(ValueError, match="external parent limit"):
        Snapshot.from_records(
            [],
            complete=True,
            external_parents=repeat("outside", 10),
            limits=EvidenceLimits(max_spans=2),
        )


def test_predicate_generator_is_stopped_at_the_limit():
    consumed = []

    def predicates():
        for index in range(70):
            consumed.append(index)
            yield Predicate("/x", "exists", True)

    with pytest.raises(ValueError, match="64 predicates"):
        Selector(predicates=predicates())
    assert len(consumed) == 65


@pytest.mark.parametrize(
    "make",
    [
        lambda: Budget("b", Selector(), "/cost", 10**1000),
        lambda: Prerequisite("p", Selector(), Selector(), max_age=10**1000),
    ],
)
def test_huge_numeric_rule_limits_are_validation_errors(make):
    with pytest.raises(ValueError, match="finite"):
        make()


def test_oversized_payload_is_rejected_before_json_encoding(monkeypatch):
    from deepeval.metrics.community.execution_contract import evidence

    def should_not_encode(*args, **kwargs):
        raise AssertionError("oversized input reached JSON encoder")

    with monkeypatch.context() as patch:
        patch.setattr(evidence.json, "dumps", should_not_encode)
        with pytest.raises(ValueError, match="byte limit"):
            evidence.canonical_json(
                {"value": "x" * 1000}, EvidenceLimits(max_payload_bytes=100)
            )


def test_native_external_boundary_generator_is_bounded():
    from deepeval.metrics.community.execution_contract.adapters import (
        snapshot_from_roots,
    )

    with pytest.raises(ValueError, match="external parent limit"):
        snapshot_from_roots(
            [],
            complete=True,
            external_parents=repeat("outside", 10),
            limits=EvidenceLimits(max_spans=2),
        )


def test_large_structural_predicate_comparison_consumes_work_budget():
    from deepeval.metrics.community.execution_contract import (
        Contract,
        EvaluationLimits,
        Occurrence,
        evaluate,
    )

    value = list(range(1000))
    snapshot = Snapshot.from_records(
        [
            {
                "id": "one",
                "kind": "tool",
                "name": "call",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {"data": value},
            },
        ],
        complete=True,
    )
    contract = Contract(
        (
            Occurrence(
                "match", Selector(predicates=(Predicate("/data", "eq", value),))
            ),
        ),
        limits=EvaluationLimits(max_work=100),
    )
    with pytest.raises(ValueError, match="work limit"):
        evaluate(snapshot, contract)


def test_aggregate_contract_size_checked_before_decoding_expected_values(
    monkeypatch,
):
    from deepeval.metrics.community.execution_contract import (
        Contract,
        Occurrence,
    )
    from deepeval.metrics.community.execution_contract.serialization import (
        contract_fingerprint,
    )

    predicates = tuple(Predicate("/data", "eq", "x" * 600000) for _ in range(2))
    contract = Contract((Occurrence("match", Selector(predicates=predicates)),))

    def unexpected_decode(self):
        raise AssertionError("oversized contract decoded its predicate values")

    monkeypatch.setattr(Predicate, "expected", property(unexpected_decode))
    with pytest.raises(ValueError, match="contract.*byte limit"):
        contract_fingerprint(contract)


def test_explicit_evidence_limits_survive_metric_and_replay_boundaries(
    tmp_path,
):
    from deepeval.metrics.community.execution_contract import (
        Contract,
        Occurrence,
        evaluate,
    )
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )

    limits = EvidenceLimits(max_depth=140)
    records = [
        {
            "id": str(i),
            "parent_id": str(i - 1) if i else None,
            "kind": "tool",
            "name": "call",
            "status": "success",
            "start": 1,
            "end": 2,
        }
        for i in range(130)
    ]
    snapshot = Snapshot.from_records(records, complete=True, limits=limits)
    contract = Contract((Occurrence("calls", Selector(), minimum=130),))
    assert evaluate(snapshot, contract, evidence_limits=limits).success
    path = tmp_path / "deep.json"
    write_artifact(path, snapshot, evidence_limits=limits)
    assert read_artifact(path, evidence_limits=limits) == snapshot
    with pytest.raises(ValueError, match="depth limit"):
        read_artifact(path)


def test_explicit_report_limits_survive_export_and_comparison(tmp_path):
    from deepeval.metrics.community.execution_contract import (
        Contract,
        EvaluationLimits,
        evaluate,
    )
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )
    from deepeval.metrics.community.execution_contract.comparison import (
        compare_reports,
    )

    limits = EvaluationLimits(max_decisions=110)
    snapshot = Snapshot.from_records(
        [
            {
                "id": str(i),
                "kind": "tool",
                "name": "pay",
                "status": "success",
                "start": 1,
                "end": 2,
            }
            for i in range(105)
        ],
        complete=True,
        clock_domain="clock",
    )
    contract = Contract(
        (Prerequisite("auth", Selector(name="pay"), Selector(name="verify")),),
        limits=limits,
    )
    report = evaluate(snapshot, contract)
    path = tmp_path / "report.json"
    write_artifact(path, report, evaluation_limits=limits)
    assert read_artifact(path, evaluation_limits=limits) == report
    assert not compare_reports(report, report, limits=limits).regressed
