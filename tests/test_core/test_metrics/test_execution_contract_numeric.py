from deepeval.metrics.community.execution_contract import (
    Budget,
    Contract,
    Prerequisite,
    Selector,
    Snapshot,
    evaluate,
)


def test_integer_timestamps_above_float_precision_are_not_reordered():
    n = 2**53
    evidence = Snapshot.from_records(
        [
            {
                "id": "verify",
                "kind": "tool",
                "name": "verify",
                "status": "success",
                "start": n - 2,
                "end": n + 1,
            },
            {
                "id": "pay",
                "kind": "tool",
                "name": "pay",
                "status": "success",
                "start": n,
                "end": n + 2,
            },
        ],
        complete=True,
        clock_domain="nanoseconds",
    )
    rule = Prerequisite(
        "authorization", Selector(name="pay"), Selector(name="verify")
    )
    assert evaluate(evidence, Contract((rule,))).status.value == "fail"


def test_integer_budget_does_not_round_away_one_unit_violation():
    n = 2**53
    evidence = Snapshot.from_records(
        [
            {
                "id": "one",
                "kind": "tool",
                "name": "call",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {"tokens": n + 1},
            },
        ],
        complete=True,
    )
    assert (
        evaluate(
            evidence, Contract((Budget("tokens", Selector(), "/tokens", n),))
        ).status.value
        == "fail"
    )


def test_mixed_budget_keeps_small_fraction_alongside_large_integer():
    n = 2**53
    evidence = Snapshot.from_records(
        [
            {
                "id": str(index),
                "kind": "tool",
                "name": "call",
                "status": "success",
                "start": 1,
                "end": 2,
                "payload": {"cost": value},
            }
            for index, value in enumerate((n, 0.25))
        ],
        complete=True,
    )
    assert (
        evaluate(
            evidence, Contract((Budget("cost", Selector(), "/cost", n),))
        ).status.value
        == "fail"
    )


def test_mixed_timestamp_age_and_recovery_deadline_keep_small_elapsed_time():
    from deepeval.metrics.community.execution_contract import Recovery

    n = 2**53
    evidence = Snapshot.from_records(
        [
            {
                "id": "verify",
                "kind": "tool",
                "name": "verify",
                "status": "error",
                "start": n,
                "end": n + 3,
                "payload": {"id": "op"},
            },
            {
                "id": "pay",
                "kind": "tool",
                "name": "pay",
                "status": "success",
                "start": float(n + 4),
                "end": float(n + 4),
                "payload": {"id": "op"},
            },
        ],
        complete=True,
        clock_domain="clock",
    )
    rules = (
        Prerequisite(
            "fresh", Selector(name="pay"), Selector(name="verify"), max_age=0
        ),
        Recovery(
            "repair",
            Selector(name="verify"),
            Selector(name="pay"),
            bindings=(("/id", "/id"),),
            within=0,
        ),
    )
    report = evaluate(evidence, Contract(rules))
    assert [rule.status.value for rule in report.rules] == ["fail", "fail"]
