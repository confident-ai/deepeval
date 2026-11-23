import asyncio
import json

from deepeval.classifiers import Classifier, Label
from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class ScriptedModel(DeepEvalBaseLLM):
    def __init__(self, label, same_as=None):
        self.label = label
        self.same_as = same_as
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        if "same underlying problem" in prompt:
            return json.dumps({"same_as": self.same_as, "reason": "r"})
        return json.dumps(
            {"label": self.label, "reason": "r", "new_label_description": f"{self.label} cases."}
        )

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "scripted"


class WordEmbedding(DeepEvalBaseEmbeddingModel):
    VOCAB = ["refund", "money", "back", "upgrade", "plan", "slow", "reply"]

    def load_model(self):
        return self

    def embed_text(self, text):
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [1.0 if word in text.lower() else 0.0 for word in self.VOCAB] for text in texts
        ]

    async def a_embed_text(self, text):
        return self.embed_text(text)

    async def a_embed_texts(self, texts):
        return self.embed_texts(texts)

    def get_model_name(self):
        return "words"


CASE = LLMTestCase(input="I want my money back")


def _classifier(model, **kwargs):
    return Classifier(
        name="topics",
        labels=[Label("Pricing", "Asks a price.")],
        model=model,
        async_mode=False,
        suggest_labels=True,
        **kwargs,
    )


def test_new_label_is_remembered_for_later_calls():
    classifier = _classifier(ScriptedModel("Refund Request"))
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.new_label_description == "Refund Request cases."
    assert [label.name for label in classifier.candidate_labels] == ["Refund Request"]

    classifier.model.label = "Plan Upgrade"
    classifier.classify(CASE, _show_indicator=False)
    assert "Candidate Labels:\n- Refund Request" in classifier.model.prompts[-2]


def test_judge_merges_a_near_duplicate_into_a_candidate():
    model = ScriptedModel("Money Back Request", same_as="Refund Request")
    classifier = _classifier(model, candidate_labels=[Label("Refund Request", "Asks for a refund.")])
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Refund Request"
    assert classifier.new_label_description == "Asks for a refund."
    assert len(classifier.candidate_labels) == 1


def test_judge_can_merge_into_an_existing_label():
    model = ScriptedModel("Cost Question", same_as="Pricing")
    classifier = _classifier(model)
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Pricing" and classifier.new_label_description is None
    assert classifier.candidate_labels == []


def test_close_embeddings_merge_without_a_judge_call():
    model = ScriptedModel("Refund Money Back")
    classifier = _classifier(
        model,
        candidate_labels=[Label("Money Back Refund", "Asks for a refund.")],
        embedding_model=WordEmbedding(),
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Money Back Refund"
    assert not any("same underlying problem" in p for p in model.prompts)


def test_copies_share_one_pool_and_converge_concurrently():
    model = ScriptedModel("Refund Request")
    classifier = Classifier(
        name="topics", labels=["Pricing"], model=model, suggest_labels=True
    )
    copies = [classifier.copy() for _ in range(5)]

    async def run():
        await asyncio.gather(*(c.a_classify(CASE, _show_indicator=False) for c in copies))

    asyncio.run(run())
    assert {c.label for c in copies} == {"Refund Request"}
    assert [label.name for label in classifier.candidate_labels] == ["Refund Request"]
