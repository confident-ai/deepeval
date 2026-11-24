import asyncio

from deepeval.classifiers import Classifier, Label, LabelDiscoveryConfig
from deepeval.test_case import LLMTestCase
from tests.test_classifiers.fakes import ReplyModel


class ScriptedModel(ReplyModel):
    def __init__(self, label, same_as=None):
        super().__init__()
        self.label = label
        self.same_as = same_as

    def respond(self, prompt):
        if "newly proposed classifier label" in prompt:
            return {"same_as": self.same_as, "reason": "r"}
        return {
            "label": self.label,
            "reason": "r",
            "new_label_description": f"{self.label} cases.",
        }


CASE = LLMTestCase(input="I want my money back")


class MultiProposalModel(ReplyModel):
    def respond(self, prompt):
        if "newly proposed classifier label" in prompt:
            same_as = (
                "Refund Request" if '"name": "Money Back"' in prompt else None
            )
            return {"same_as": same_as, "reason": "r"}
        return {
            "labels": [
                {
                    "label": "Shipping Delay",
                    "reason": "r",
                    "new_label_description": "Order arrives late.",
                },
                {
                    "label": "Money Back",
                    "reason": "r",
                    "new_label_description": "Wants a refund.",
                },
            ]
        }


def _classifier(model, **config):
    return Classifier(
        name="topics",
        labels=[Label("Pricing", "Asks a price.")],
        model=model,
        async_mode=False,
        label_discovery_config=LabelDiscoveryConfig(**config),
    )


def test_new_label_is_remembered_for_later_calls():
    classifier = _classifier(ScriptedModel("Refund Request"))
    classifier.classify(CASE, _show_indicator=False)
    assert (
        classifier.classifications[0].new_label_description
        == "Refund Request cases."
    )
    assert [label.name for label in classifier.candidate_labels] == [
        "Refund Request"
    ]

    classifier.model.label = "Plan Upgrade"
    classifier.classify(CASE, _show_indicator=False)
    assert "Candidate Labels:\n- Refund Request" in classifier.model.prompts[-2]


def test_judge_merges_a_near_duplicate_into_a_candidate():
    model = ScriptedModel("Money Back Request", same_as="Refund Request")
    classifier = _classifier(
        model, candidate_labels=[Label("Refund Request", "Asks for a refund.")]
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Refund Request"
    assert (
        classifier.classifications[0].new_label_description
        == "Asks for a refund."
    )
    assert len(classifier.candidate_labels) == 1


def test_judge_can_merge_into_an_existing_label():
    model = ScriptedModel("Cost Question", same_as="Pricing")
    classifier = _classifier(model)
    classifier.classify(CASE, _show_indicator=False)
    assert (
        classifier.label == "Pricing"
        and classifier.classifications[0].new_label_description is None
    )
    assert classifier.candidate_labels == []


def test_close_embeddings_merge_without_a_judge_call():
    model = ScriptedModel("Refund Money Back")
    classifier = _classifier(
        model,
        candidate_labels=[Label("Money Back Refund", "Asks for a refund.")],
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Money Back Refund"
    assert not any(
        "newly proposed classifier label" in p for p in model.prompts
    )


def test_copies_share_one_pool_and_converge_concurrently():
    model = ScriptedModel("Refund Request")
    classifier = Classifier(
        name="topics",
        labels=["Pricing"],
        model=model,
        label_discovery_config=LabelDiscoveryConfig(),
    )
    copies = [classifier.copy() for _ in range(5)]

    async def run():
        await asyncio.gather(
            *(c.a_classify(CASE, _show_indicator=False) for c in copies)
        )

    asyncio.run(run())
    assert {c.label for c in copies} == {"Refund Request"}
    assert [label.name for label in classifier.candidate_labels] == [
        "Refund Request"
    ]


def test_a_proposal_past_the_cap_can_still_merge():
    classifier = Classifier(
        name="topics",
        labels=["Pricing"],
        model=MultiProposalModel(),
        async_mode=False,
        multi_label=True,
        label_discovery_config=LabelDiscoveryConfig(
            candidate_labels=[Label("Refund Request", "Asks for a refund.")],
            max_new_labels=1,
        ),
    )
    classifier.classify(CASE, _show_indicator=False)
    assert [c.label for c in classifier.classifications] == [
        "Shipping Delay",
        "Refund Request",
    ]
    assert [label.name for label in classifier.candidate_labels] == [
        "Refund Request",
        "Shipping Delay",
    ]
    assert classifier.steps.label_discovery.merged_labels == {
        "Shipping Delay": "Shipping Delay",
        "Money Back": "Refund Request",
    }


class UnreadableMergeModel(ScriptedModel):
    def respond(self, prompt):
        if "newly proposed classifier label" in prompt:
            return "not json"
        return super().respond(prompt)


def test_an_unreadable_merge_answer_keeps_the_label_as_new():
    classifier = _classifier(
        UnreadableMergeModel("Money Back Request"),
        candidate_labels=[Label("Shipping Delay", "Order arrives late.")],
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == "Money Back Request"
    assert classifier.error is None
    assert [label.name for label in classifier.candidate_labels] == [
        "Shipping Delay",
        "Money Back Request",
    ]
