import pytest


def test_missing_is_not_json_null_and_pointer_escapes_are_supported():
    from deepeval.metrics.community.execution_contract.selectors import (
        MISSING,
        resolve_pointer,
    )

    value = {"a/b": {"~key": [None, False, 0]}}
    assert resolve_pointer(value, "/a~1b/~0key/0") is None
    assert resolve_pointer(value, "/a~1b/~0key/1") is False
    assert resolve_pointer(value, "/a~1b/~0key/3") is MISSING
    assert resolve_pointer(value, "") is value


@pytest.mark.parametrize("pointer", ["a", "/~2", "/a~", "/a~x"])
def test_invalid_pointer_is_configuration_error(pointer):
    from deepeval.metrics.community.execution_contract.selectors import (
        resolve_pointer,
    )

    with pytest.raises(ValueError):
        resolve_pointer({}, pointer)


def test_array_pointer_never_accepts_negative_or_noncanonical_index():
    from deepeval.metrics.community.execution_contract.selectors import (
        MISSING,
        resolve_pointer,
    )

    for pointer in (
        "/-1",
        "/01",
        "/+1",
        "/1.0",
        "/-",
        "/999999999999999999999999",
    ):
        assert resolve_pointer(["a", "b"], pointer) is MISSING


def test_json_matching_preserves_nested_types():
    from deepeval.metrics.community.execution_contract.selectors import (
        json_equal,
    )

    assert not json_equal(True, 1)
    assert not json_equal({"a": [False]}, {"a": [0]})
    assert json_equal({"b": 2, "a": [1]}, {"a": [1], "b": 2})


def test_selector_missing_value_is_unknown_but_name_mismatch_is_no():
    from deepeval.metrics.community.execution_contract.evidence import Snapshot
    from deepeval.metrics.community.execution_contract.selectors import (
        Predicate,
        Selector,
        Match,
    )

    span = Snapshot.from_records(
        [
            {
                "id": "a",
                "kind": "tool",
                "name": "pay",
                "status": "success",
                "start": 1,
                "end": 2,
            }
        ],
        complete=True,
    ).spans[0]
    selected = Selector(
        name="pay", predicates=(Predicate("/input/account", "eq", "alice"),)
    )
    assert selected.matches(span) is Match.UNKNOWN
    assert (
        Selector(name="verify", predicates=selected.predicates).matches(span)
        is Match.NO
    )
    assert Selector(name="pay").matches(span) is Match.YES


def test_binding_missing_null_and_type_differences_never_match():
    from deepeval.metrics.community.execution_contract.selectors import (
        Match,
        match_bindings,
    )

    bindings = (("/account", "/account"),)
    assert match_bindings({}, {"account": None}, bindings) is Match.UNKNOWN
    assert (
        match_bindings({"account": False}, {"account": 0}, bindings) is Match.NO
    )
    assert (
        match_bindings({"account": "a"}, {"account": "a"}, bindings)
        is Match.YES
    )


def test_exists_predicate_explicitly_tests_absence():
    from deepeval.metrics.community.execution_contract.selectors import (
        Predicate,
        Match,
    )

    assert Predicate("/x", "exists", False).matches({}) is Match.YES
    assert Predicate("/x", "exists", True).matches({"x": None}) is Match.YES


def test_predicate_detaches_expected_values():
    from deepeval.metrics.community.execution_contract.selectors import (
        Predicate,
        Match,
    )

    expected = {"x": [1]}
    predicate = Predicate("/value", "eq", expected)
    expected["x"].append(2)
    assert predicate.matches({"value": {"x": [1]}}) is Match.YES
