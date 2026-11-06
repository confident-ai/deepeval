import pytest

from deepeval.metrics.community.execution_contract import (
    Contract,
    EvaluationLimits,
    Prerequisite,
    Selector,
    Snapshot,
    evaluate,
)


def records(count):
    for index in range(count):
        for name, start in (("verify", 1), ("pay", 3)):
            yield {
                "id": f"{name}-{index}",
                "kind": "tool",
                "name": name,
                "status": "success",
                "start": start,
                "end": start + 1,
                "payload": {"account": index},
            }


def test_large_subject_bound_join_avoids_cross_product_work():
    evidence = Snapshot.from_records(
        records(2000), complete=True, clock_domain="clock"
    )
    contract = Contract(
        (
            Prerequisite(
                "authorized",
                Selector(name="pay"),
                Selector(name="verify"),
                bindings=(("/account", "/account"),),
            ),
        ),
        limits=EvaluationLimits(max_work=60000),
    )
    assert evaluate(evidence, contract).success


@pytest.mark.parametrize(
    "left,right,matched",
    [
        (True, 1, False),
        (1, 1.0, False),
        (-0.0, 0.0, True),
        (None, None, True),
        ({"b": [1], "a": False}, {"a": False, "b": [1]}, True),
    ],
)
def test_join_index_preserves_typed_json_equality(left, right, matched):
    from deepeval.metrics.community.execution_contract.index import BindingIndex

    evidence = Snapshot.from_records(records(1), complete=True)
    target, candidate = evidence.spans
    payloads = {target.id: {"v": left}, candidate.id: {"v": right}}
    index = BindingIndex(
        [candidate], payloads, (("/v", "/v"),), lambda amount=1: None
    )
    assert bool(list(index.candidates(target))) is matched


def test_missing_join_fields_remain_possible_but_known_mismatch_excludes():
    from deepeval.metrics.community.execution_contract.index import BindingIndex

    evidence = Snapshot.from_records(records(1), complete=True)
    target, candidate = evidence.spans
    bindings = (("/account", "/account"), ("/operation", "/operation"))
    for account, expected in (("alice", True), ("bob", False)):
        payloads = {
            target.id: {"account": "alice", "operation": "p1"},
            candidate.id: {"account": account},
        }
        index = BindingIndex(
            [candidate], payloads, bindings, lambda amount=1: None
        )
        assert bool(list(index.candidates(target))) is expected
