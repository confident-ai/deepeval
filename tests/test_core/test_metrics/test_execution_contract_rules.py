import pytest

from deepeval.metrics.community.execution_contract.evidence import Snapshot
from deepeval.metrics.community.execution_contract.selectors import Selector


def event(id, name, start=0, end=1, account="alice", **extra):
    return dict(
        id=id,
        kind="tool",
        name=name,
        status="success",
        start=start,
        end=end,
        payload={"account": account},
        **extra
    )


def run(records, rule, complete=True, clock="local"):
    from deepeval.metrics.community.execution_contract.engine import evaluate
    from deepeval.metrics.community.execution_contract.rules import Contract

    return evaluate(
        Snapshot.from_records(records, complete=complete, clock_domain=clock),
        Contract((rule,)),
    )


def prerequisite(**options):
    from deepeval.metrics.community.execution_contract.rules import Prerequisite

    return Prerequisite(
        "auth",
        target=Selector(name="pay"),
        prerequisite=Selector(name="verify", status="success"),
        bindings=(("/account", "/account"),),
        **options
    )


@pytest.mark.parametrize(
    "complete,minimum,maximum,expected",
    [
        (True, 1, 2, "pass"),
        (True, 2, 3, "fail"),
        (False, 1, None, "pass"),
        (False, 1, 2, "unknown"),
        (False, 2, 3, "unknown"),
        (False, 0, 0, "fail"),
    ],
)
def test_occurrence_bounds_respect_partial_capture(
    complete, minimum, maximum, expected
):
    from deepeval.metrics.community.execution_contract.rules import Occurrence

    result = run(
        [event("a", "pay")],
        Occurrence(
            "count", Selector(name="pay"), minimum=minimum, maximum=maximum
        ),
        complete=complete,
    )
    assert result.status.value == expected


@pytest.mark.parametrize(
    "auth,payment,expected",
    [
        (event("v", "verify", 0, 1), event("p", "pay", 1, 2), "pass"),
        (event("v", "verify", 0, 2), event("p", "pay", 1, 3), "fail"),
        (event("v", "verify", 3, 4), event("p", "pay", 1, 2), "fail"),
        (
            event("v", "verify", 0, 1, account="bob"),
            event("p", "pay", 2, 3),
            "fail",
        ),
        (
            event("v", "verify", 0, 1, account=False),
            event("p", "pay", 2, 3, account=0),
            "fail",
        ),
    ],
)
def test_prerequisite_requires_same_subject_and_completed_order(
    auth, payment, expected
):
    report = run([auth, payment], prerequisite())
    assert report.status.value == expected
    assert report.rules[0].decisions[0].target_id == "p"


def test_error_is_not_authorization():
    auth = event("v", "verify")
    auth["status"] = "error"
    assert (
        run([auth, event("p", "pay", 2, 3)], prerequisite()).status.value
        == "fail"
    )


def test_missing_subject_and_missing_clock_stay_unknown():
    auth = event("v", "verify")
    auth["payload"] = {}
    payment = event("p", "pay", 2, 3)
    assert run([auth, payment], prerequisite()).status.value == "unknown"
    auth["payload"] = {"account": "alice"}
    assert (
        run([auth, payment], prerequisite(), clock=None).status.value
        == "unknown"
    )


def test_partial_capture_cannot_prove_missing_prerequisite():
    assert (
        run([event("p", "pay")], prerequisite(), complete=False).status.value
        == "unknown"
    )


def test_no_target_is_explicit_vacuous_pass_only_on_complete_capture():
    report = run([], prerequisite())
    assert report.status.value == "pass"
    assert report.rules[0].reason_code == "no_targets"
    assert run([], prerequisite(), complete=False).status.value == "unknown"


def test_revocation_after_verification_blocks_payment_and_new_verification_restores():
    rule = prerequisite(invalidated_by=Selector(name="revoke"))
    records = [
        event("v", "verify", 0, 1),
        event("r", "revoke", 2, 3),
        event("p", "pay", 6, 7),
    ]
    assert run(records, rule).status.value == "fail"
    assert (
        run(records + [event("v2", "verify", 4, 5)], rule).status.value
        == "pass"
    )


def test_equal_time_revocation_is_ambiguous():
    rule = prerequisite(invalidated_by=Selector(name="revoke"))
    records = [
        event("v", "verify", 0, 1),
        event("r", "revoke", 0, 1),
        event("p", "pay", 2, 3),
    ]
    assert run(records, rule).status.value == "unknown"


def test_expired_verification_does_not_authorize():
    rule = prerequisite(max_age=2)
    assert (
        run([event("v", "verify"), event("p", "pay", 4, 5)], rule).status.value
        == "fail"
    )


def test_permutation_is_deterministic_and_reports_do_not_copy_payload():
    rule = prerequisite()
    records = [event("v", "verify"), event("p", "pay", 2, 3)]
    left, right = run(records, rule), run(list(reversed(records)), rule)
    assert left.to_dict() == right.to_dict()
    assert "alice" not in str(left.to_dict())


def test_witness_limit_preserves_aggregate_failure():
    from deepeval.metrics.community.execution_contract.engine import evaluate
    from deepeval.metrics.community.execution_contract.rules import (
        Contract,
        EvaluationLimits,
    )

    records = [event(str(i), "pay", i, i + 1) for i in range(10)]
    contract = Contract(
        (prerequisite(),), limits=EvaluationLimits(max_decisions=2)
    )
    report = evaluate(
        Snapshot.from_records(records, complete=True, clock_domain="local"),
        contract,
    )
    assert report.status.value == "fail"
    assert len(report.rules[0].decisions) == 2
    assert report.rules[0].decision_count == 10
    assert report.rules[0].omitted_decisions == 8


def test_comparison_budget_raises_instead_of_partial_pass():
    from deepeval.metrics.community.execution_contract.engine import evaluate
    from deepeval.metrics.community.execution_contract.rules import (
        Contract,
        EvaluationLimits,
    )

    records = [event("v", "verify"), event("p", "pay", 2, 3)]
    with pytest.raises(ValueError, match="work limit"):
        evaluate(
            Snapshot.from_records(records, complete=True, clock_domain="local"),
            Contract((prerequisite(),), limits=EvaluationLimits(max_work=1)),
        )


def test_configuration_is_validated_before_any_evidence_is_matched():
    from deepeval.metrics.community.execution_contract.rules import (
        Contract,
        Occurrence,
        Prerequisite,
    )

    with pytest.raises(ValueError):
        Prerequisite(
            "bad", Selector(), Selector(), bindings=(("bad-pointer", "/x"),)
        )
    with pytest.raises(ValueError):
        Occurrence("bad", Selector(), minimum=2, maximum=1)
    with pytest.raises(ValueError):
        Contract((prerequisite(), prerequisite()))


def test_required_unknown_cannot_be_averaged_into_success():
    from deepeval.metrics.community.execution_contract.engine import evaluate
    from deepeval.metrics.community.execution_contract.rules import (
        Contract,
        Occurrence,
    )

    snap = Snapshot.from_records(
        [event("p", "pay")], complete=False, clock_domain="local"
    )
    report = evaluate(
        snap,
        Contract((Occurrence("present", Selector(name="pay")), prerequisite())),
    )
    assert report.status.value == "unknown"
    assert not report.success


def recovery(**kwargs):
    from deepeval.metrics.community.execution_contract.rules import Recovery

    return Recovery(
        "recover",
        Selector(name="pay", status="error"),
        Selector(name="refund", status="success"),
        bindings=(("/account", "/account"),),
        **kwargs
    )


def test_recovery_matches_subject_and_completes_before_continuation():
    failure = event("f", "pay", 0, 1)
    failure["status"] = "error"
    repair = event("r", "refund", 2, 3)
    continuation = event("c", "ship", 4, 5)
    rule = recovery(before=Selector(name="ship"))
    assert run([failure, repair, continuation], rule).success
    repair["end"] = 4.5
    assert run([failure, repair, continuation], rule).status.value == "fail"
    repair["end"] = 3
    repair["payload"]["account"] = "bob"
    assert run([failure, repair, continuation], rule).status.value == "fail"


def test_recovery_requires_fresh_attempt_and_observed_deadline():
    failure = event("f", "pay", 2, 3)
    failure["status"] = "error"
    assert (
        run([failure, event("r", "refund", 0, 1)], recovery()).status.value
        == "fail"
    )
    assert (
        run(
            [failure, event("r", "refund", 4, 8)], recovery(within=2)
        ).status.value
        == "fail"
    )
    assert run([failure], recovery(), complete=False).status.value == "unknown"


@pytest.mark.parametrize(
    "cost,complete,expected",
    [
        (1, True, "pass"),
        (3, True, "fail"),
        (1, False, "unknown"),
        (3, False, "fail"),
        (None, True, "unknown"),
    ],
)
def test_resource_budget_keeps_missing_and_partial_distinct(
    cost, complete, expected
):
    from deepeval.metrics.community.execution_contract.rules import Budget

    row = event("a", "model")
    row["payload"] = {"cost": cost}
    rule = Budget("cost", Selector(name="model"), "/cost", maximum=2)
    assert run([row], rule, complete=complete).status.value == expected


def test_resource_budget_never_sums_parent_aggregate_and_child():
    from deepeval.metrics.community.execution_contract.rules import Budget

    parent = event("a", "model")
    child = event("b", "model", parent_id="a")
    parent["payload"] = child["payload"] = {"cost": 1}
    assert (
        run(
            [parent, child],
            Budget("cost", Selector(name="model"), "/cost", maximum=5),
        ).status.value
        == "unknown"
    )
    assert run(
        [parent, child],
        Budget(
            "cost", Selector(name="model"), "/cost", maximum=5, exclusive=True
        ),
    ).success


@pytest.mark.parametrize("value", [-1, False, "2"])
def test_invalid_resource_values_are_evidence_errors(value):
    from deepeval.metrics.community.execution_contract.rules import Budget

    row = event("a", "model")
    row["payload"] = {"cost": value}
    with pytest.raises(ValueError, match="resource"):
        run([row], Budget("cost", Selector(), "/cost", maximum=5))
