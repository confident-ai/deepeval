"""The test-case setter must preserve the dataset's single-/multi-turn invariant."""

import pytest

from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn


def case(multi):
    return (
        ConversationalTestCase(turns=[Turn(role="user", content="hi")])
        if multi
        else LLMTestCase(input="hi")
    )


def golden(multi):
    return (
        ConversationalGolden(scenario="say hello")
        if multi
        else Golden(input="hi")
    )


@pytest.mark.parametrize("multi", [False, True])
def test_empty_dataset_assignment(multi):
    ds = EvaluationDataset()
    item = case(multi)
    ds.test_cases = [item]
    assert ds.test_cases == [item]
    assert ds._multi_turn is multi


@pytest.mark.parametrize("multi", [False, True])
def test_empty_assignment_retains_goldens(multi):
    item = golden(multi)
    ds = EvaluationDataset(goldens=[item])
    ds.test_cases = []
    assert ds.test_cases == []
    assert ds.goldens == [item]
    assert ds._multi_turn is multi


@pytest.mark.parametrize("multi", [False, True])
def test_replacement_can_switch_when_no_goldens(multi):
    ds = EvaluationDataset()
    ds.add_test_case(case(not multi))
    item = case(multi)
    ds.test_cases = [item]
    assert ds.test_cases == [item]
    assert ds._multi_turn is multi


@pytest.mark.parametrize("multi", [False, True])
def test_incompatible_goldens_rejected_without_data_loss(multi):
    g = golden(multi)
    ds = EvaluationDataset(goldens=[g])
    old = case(multi)
    ds.add_test_case(old)
    with pytest.raises(TypeError):
        ds.test_cases = [case(not multi)]
    assert ds.goldens == [g]
    assert ds.test_cases == [old]


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_replacement_rejected_without_data_loss(multi, reverse):
    ds = EvaluationDataset()
    old = case(multi)
    ds.add_test_case(old)
    incoming = [case(False), case(True)]
    if reverse:
        incoming.reverse()
    with pytest.raises(TypeError):
        ds.test_cases = incoming
    assert ds.test_cases == [old]


@pytest.mark.parametrize("multi", [False, True])
def test_matching_goldens_retained(multi):
    g = golden(multi)
    ds = EvaluationDataset(goldens=[g])
    item = case(multi)
    ds.test_cases = [item]
    assert ds.test_cases == [item]
    assert ds.goldens == [g]


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("items", [[], [None, object()]])
def test_empty_or_ignored_replacement_retains_mode(multi, items):
    ds = EvaluationDataset()
    ds.add_test_case(case(multi))
    ds.test_cases = items
    assert ds.test_cases == []
    assert ds._multi_turn is multi


@pytest.mark.parametrize("multi", [False, True])
def test_success_preserves_dataset_metadata_and_ranks(multi):
    ds = EvaluationDataset()
    ds._alias, ds._id, ds._version = "dataset", "id", "version"
    incoming = [case(multi), case(multi)]
    ds.test_cases = [None, incoming[0], object(), incoming[1]]
    assert ds.test_cases == incoming
    for rank, item in enumerate(incoming):
        assert item._dataset_alias == "dataset"
        assert item._dataset_id == "id"
        assert item._dataset_version == "version"
        assert item._dataset_rank == rank


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_rejected_assignment_does_not_mutate_incoming_metadata(multi, mixed):
    ds = EvaluationDataset(goldens=[golden(multi)])
    ds._alias, ds._id, ds._version = "target", "target-id", "target-version"
    incoming = [case(not multi)]
    if mixed:
        incoming.append(case(multi))
    for item in incoming:
        item._dataset_alias = "original"
        item._dataset_id = "original-id"
        item._dataset_version = "original-version"
        item._dataset_rank = 42
    with pytest.raises(TypeError):
        ds.test_cases = incoming
    for item in incoming:
        assert item._dataset_alias == "original"
        assert item._dataset_id == "original-id"
        assert item._dataset_version == "original-version"
        assert item._dataset_rank == 42


def test_assigned_conversation_can_be_exported(tmp_path):
    import json
    from pathlib import Path

    ds = EvaluationDataset()
    item = ConversationalTestCase(
        scenario="Say hello", turns=[Turn(role="user", content="hi")]
    )
    ds.test_cases = [item]
    path = ds.save_as("json", str(tmp_path), include_test_cases=True)
    data = json.loads(Path(path).read_text())
    assert len(data) == 1
    assert data[0]["scenario"] == "Say hello"
    assert data[0]["turns"][0]["role"] == "user"
    assert data[0]["turns"][0]["content"] == "hi"
