"""Keep classifier expectations when datasets are stored locally."""

import csv
import json
import socket

import pytest

from deepeval import evaluate
from deepeval.classifiers.base_classifier import BaseClassifier, Label
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.dataset.utils import (
    convert_goldens_to_test_cases,
    convert_convo_goldens_to_convo_test_cases,
)
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric, BaseConversationalMetric
from deepeval.test_case import Turn


LABELS = {
    "refusal": "refused",
    "toneCheck": "polite",
    "中文分类": "安全",
    'comma,pipe|quote"': 'label,pipe|quote"',
}
MISSING = object()
LOADERS = [
    (kind, multi_turn, as_test_cases)
    for kind in ("json", "jsonl", "csv")
    for multi_turn, as_test_cases in (
        (False, False),
        (True, False),
        (False, True),
    )
    if not (kind == "jsonl" and as_test_cases)
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("These dataset tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)


def make_golden(multi_turn, labels):
    if multi_turn:
        return ConversationalGolden(
            scenario="Handle a request",
            turns=[
                Turn(role="user", content="Hi"),
                Turn(role="assistant", content="I answered"),
            ],
            expected_labels=labels,
        )
    return Golden(
        input="Hi", actual_output="I answered", expected_labels=labels
    )


def to_test_cases(goldens, multi_turn):
    converter = (
        convert_convo_goldens_to_convo_test_cases
        if multi_turn
        else convert_goldens_to_test_cases
    )
    return converter(goldens)


def load_file(path, kind, as_test_cases=False):
    dataset = EvaluationDataset()
    if as_test_cases:
        kwargs = (
            {
                "input_col_name": "input",
                "actual_output_col_name": "actual_output",
            }
            if kind == "csv"
            else {
                "input_key_name": "input",
                "actual_output_key_name": "actual_output",
            }
        )
        getattr(dataset, f"add_test_cases_from_{kind}_file")(
            str(path), **kwargs
        )
        return dataset.test_cases[0]
    getattr(dataset, f"add_goldens_from_{kind}_file")(str(path))
    return dataset.goldens[0]


def write_external_file(tmp_path, kind, multi_turn, labels):
    record = (
        {
            "scenario": "Handle a request",
            "turns": [{"role": "user", "content": "Hi"}],
        }
        if multi_turn
        else {"input": "Hi", "actual_output": "I answered"}
    )
    if labels is not MISSING:
        record["expected_labels"] = labels
    path = tmp_path / ("external." + kind)
    if kind == "json":
        path.write_text(json.dumps([record]), encoding="utf-8")
    elif kind == "jsonl":
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if key in ("turns", "expected_labels")
                        else value
                    )
                    for key, value in record.items()
                }
            )
    return path


@pytest.mark.parametrize("kind", ["json", "jsonl", "csv"])
@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("labels", [LABELS, {}, None])
@pytest.mark.parametrize("include_test_cases", [False, True])
def test_expected_labels_roundtrip(
    tmp_path, kind, multi_turn, labels, include_test_cases
):
    original = make_golden(multi_turn, labels)
    dataset = EvaluationDataset([] if include_test_cases else [original])
    if include_test_cases:
        dataset.add_test_case(to_test_cases([original], multi_turn)[0])
    path = dataset.save_as(
        kind,
        str(tmp_path),
        file_name="roundtrip",
        include_test_cases=include_test_cases,
    )
    assert load_file(path, kind).expected_labels == labels
    assert original.expected_labels == labels


@pytest.mark.parametrize("kind,multi_turn,as_test_cases", LOADERS)
@pytest.mark.parametrize("labels", [LABELS, {}, None, MISSING])
def test_import_expected_labels(
    tmp_path, kind, multi_turn, as_test_cases, labels
):
    path = write_external_file(tmp_path, kind, multi_turn, labels)
    expected = None if labels is MISSING else labels
    assert load_file(path, kind, as_test_cases).expected_labels == expected


@pytest.mark.parametrize("kind,multi_turn,as_test_cases", LOADERS)
@pytest.mark.parametrize(
    "labels", [[], "refused", {"refusal": []}, 0, False, 42, True]
)
def test_invalid_expected_labels_are_not_silently_dropped(
    tmp_path, kind, multi_turn, as_test_cases, labels
):
    path = write_external_file(tmp_path, kind, multi_turn, labels)
    with pytest.raises((TypeError, ValueError), match="expected_labels"):
        load_file(path, kind, as_test_cases)


@pytest.mark.parametrize(
    "multi_turn,as_test_cases", [(False, False), (True, False), (False, True)]
)
def test_blank_csv_expected_labels_are_optional(
    tmp_path, multi_turn, as_test_cases
):
    path = write_external_file(tmp_path, "csv", multi_turn, None)
    path.write_text(
        path.read_text(encoding="utf-8").replace(",null", ","), encoding="utf-8"
    )
    assert load_file(path, "csv", as_test_cases).expected_labels is None


class AlwaysAnswered(BaseClassifier):
    name = "refusal"
    labels = [Label(name="refused"), Label(name="answered")]

    def classify(self, test_case, **kwargs):
        self.label = "answered"
        return self.label

    async def a_classify(self, test_case, **kwargs):
        return self.classify(test_case, **kwargs)


class AlwaysPassing(BaseMetric):
    threshold = 0.5

    @property
    def __name__(self):
        return "Passing metric"

    def measure(self, test_case, **kwargs):
        self.score = 1.0
        self.success = True
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case, **kwargs)


class AlwaysPassingConversation(BaseConversationalMetric):
    threshold = 0.5

    @property
    def __name__(self):
        return "Passing conversational metric"

    def measure(self, test_case, **kwargs):
        self.score = 1.0
        self.success = True
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case, **kwargs)


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("has_metric", [False, True])
def test_roundtrip_keeps_classifier_gate(
    tmp_path, run_async, multi_turn, has_metric
):
    original = make_golden(multi_turn, {"refusal": "refused"})
    path = EvaluationDataset([original]).save_as(
        "json", str(tmp_path), file_name="gate"
    )
    restored = load_file(path, "json")
    metric = AlwaysPassingConversation() if multi_turn else AlwaysPassing()
    results = evaluate(
        to_test_cases([original, restored], multi_turn),
        metrics=[metric] if has_metric else [],
        classifiers=[AlwaysAnswered()],
        async_config=AsyncConfig(run_async=run_async),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False, print_results=False, inspect_after_run=False
        ),
    ).test_results
    assert len(results) == 2
    assert all(result.success is False for result in results)
    assert all(result.classifications[0].success is False for result in results)
