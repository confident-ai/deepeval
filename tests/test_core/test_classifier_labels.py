import asyncio

import pytest

from deepeval import evaluate
from deepeval.classifiers import Classifier, Label
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class OfflineJudge(DeepEvalBaseLLM):
    def __init__(self, response_label):
        self.response_label = response_label
        self.calls = 0
        super().__init__()

    def load_model(self):
        return None

    def get_model_name(self):
        return "offline-label-fixture"

    def generate(self, prompt, schema=None):
        self.calls += 1
        return schema(label=self.response_label, reason="Local fixture")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def classify(classifier, entrypoint):
    test_case = LLMTestCase(input="Hello", actual_output="Hello")
    if entrypoint == "async":
        return asyncio.run(
            classifier.a_classify(test_case, _show_indicator=False)
        )
    return classifier.classify(test_case, _show_indicator=False)


@pytest.mark.parametrize("entrypoint", ["sync", "async"])
@pytest.mark.parametrize("label_kind", ["string", "label"])
@pytest.mark.parametrize("response_label", ["safe", "SAFE", " safe "])
def test_classifier_matches_padded_declared_labels(
    entrypoint, label_kind, response_label
):
    declared_name = " safe "
    label = (
        Label(declared_name, "A safe response")
        if label_kind == "label"
        else declared_name
    )
    model = OfflineJudge(response_label)
    classifier = Classifier(
        name="decision",
        labels=[label, "unsafe"],
        model=model,
        eval_mode="llm",
    )

    assert classify(classifier, entrypoint) == declared_name
    assert classifier.label == declared_name
    assert classifier.error is None
    assert classifier.is_successful(declared_name) is True
    assert model.calls == 1


@pytest.mark.parametrize("entrypoint", ["sync", "async"])
@pytest.mark.parametrize("response_label", ["safe", "SAFE"])
def test_classifier_keeps_unpadded_label_matching(entrypoint, response_label):
    classifier = Classifier(
        name="decision",
        labels=["safe", "unsafe"],
        model=OfflineJudge(response_label),
        eval_mode="llm",
    )

    assert classify(classifier, entrypoint) == "safe"
    assert classifier.error is None


@pytest.mark.parametrize("entrypoint", ["sync", "async"])
@pytest.mark.parametrize("allow_none", [False, True])
def test_classifier_none_sentinel_still_respects_allow_none(
    entrypoint, allow_none
):
    classifier = Classifier(
        name="decision",
        labels=[" safe ", "unsafe"],
        model=OfflineJudge(" NONE "),
        eval_mode="llm",
        allow_none=allow_none,
    )

    assert classify(classifier, entrypoint) is None
    assert (classifier.error is None) is allow_none


@pytest.mark.parametrize("entrypoint", ["sync", "async"])
def test_classifier_still_rejects_unknown_labels(entrypoint):
    classifier = Classifier(
        name="decision",
        labels=[" safe ", "unsafe"],
        model=OfflineJudge("other"),
        eval_mode="llm",
        allow_none=True,
    )

    assert classify(classifier, entrypoint) is None
    assert classifier.error is not None
    assert classifier.is_successful(" safe ") is False


@pytest.mark.parametrize("run_async", [False, True])
def test_evaluate_preserves_padded_label_and_expected_verdict(run_async):
    declared_name = " safe "
    test_case = LLMTestCase(
        input="Hello",
        actual_output="Hello",
        expected_labels={"decision": declared_name},
    )
    classifier = Classifier(
        name="decision",
        labels=[declared_name, "unsafe"],
        model=OfflineJudge("safe"),
        eval_mode="llm",
    )

    result = evaluate(
        [test_case],
        classifiers=[classifier],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(
            show_indicator=False, print_results=False, inspect_after_run=False
        ),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
    )

    test_result = result.test_results[0]
    classification = test_result.classifications[0]
    assert test_result.success is True
    assert classification.label == declared_name
    assert classification.expected_label == declared_name
    assert classification.error is None
    assert classification.success is True
