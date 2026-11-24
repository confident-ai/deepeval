import pytest

from deepeval.classifiers import (
    AgentProfile,
    Classifier,
    EvidenceConfig,
    IssueClassifier,
    Label,
    LabelDiscoveryConfig,
    RefusalClassifier,
    ReviewConfig,
    a_classify_with_decision_model,
    classify_with_decision_model,
)
from deepeval.config.eval_mode import EvalMode
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    SystemOneAnswers,
)
from deepeval.test_case import LLMTestCase
from tests.test_classifiers.fakes import ReplyModel
from tests.test_metrics.system_one_fakes import FakeSystemOneModel

LABELS = [
    Label("Billing", "Questions about invoices."),
    Label("Shipping", "Questions about deliveries."),
]
TEST_CASE = LLMTestCase(
    input="Where is my parcel?", actual_output="On its way."
)


def _choice(choice, confidence):
    return ChoiceAnswer(
        choice=choice, probabilities={choice: confidence}, confidence=confidence
    )


def _answering(choices=None, nouls=None):
    def answer(questions):
        out = SystemOneAnswers()
        for key, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                out.choices[key] = choices[key]
            else:
                out.nouls[key] = NoulAnswer(probability=nouls[key])
        return out

    return answer


def _classifier(jev, **kwargs):
    kwargs.setdefault("labels", LABELS)
    kwargs.setdefault("allow_none", True)
    return Classifier(
        name="topic",
        eval_mode="decision_model",
        decision_model=jev,
        async_mode=False,
        **kwargs,
    )


def test_choice_carries_the_classifier_and_app_in_its_instructions():
    jev = FakeSystemOneModel(
        answer_fn=_answering(choices={"0": _choice("Shipping", 0.91234)})
    )
    classifier = _classifier(
        jev,
        description="What the user asks about.",
        agent_profile=AgentProfile(description="A store's support bot."),
    )
    assert classifier.classify(TEST_CASE, _show_indicator=False) == "Shipping"
    _, questions = jev.calls[0]
    instructions = questions["0"].instructions
    assert '"topic" classifier' in instructions["question"]
    assert instructions["classifier_purpose"] == "What the user asks about."
    assert instructions["application"] == "A store's support bot."
    assert "…[truncated]" in instructions["content_format"]
    assert "NONE" in questions["0"].options
    assert classifier.reason == (
        'Decided by fake-jev. Label "Shipping" (P=0.91, confidence=0.91).'
    )
    assert classifier.classifications[0].confidence == 0.9123


def test_metadata_reaches_the_decision_model_with_its_description():
    jev = FakeSystemOneModel(
        answer_fn=_answering(choices={"0": _choice("Shipping", 0.9)})
    )
    test_case = LLMTestCase(
        input="Where is my parcel?",
        actual_output="On its way.",
        metadata={"evidence": {"listening": "The agent sounds rushed."}},
    )
    classify_with_decision_model(
        [_classifier(jev)],
        test_case,
        metadata_description='"evidence" was gathered from the call audio.',
    )
    state, questions = jev.calls[0]
    assert state["metadata"] == test_case.metadata
    assert (
        questions["0"]
        .instructions["content_format"]
        .endswith(
            "`metadata` holds context gathered outside the application: "
            '"evidence" was gathered from the call audio.'
        )
    )


def test_metadata_without_a_description_stays_out_of_the_state():
    jev = FakeSystemOneModel(
        answer_fn=_answering(choices={"0": _choice("Shipping", 0.9)})
    )
    test_case = LLMTestCase(
        input="Where is my parcel?",
        actual_output="On its way.",
        metadata={"source": "crm"},
    )
    classify_with_decision_model([_classifier(jev)], test_case)
    state, questions = jev.calls[0]
    assert "metadata" not in state
    assert "`metadata`" not in questions["0"].instructions["content_format"]


def test_none_choice_gives_no_fitting_label_reason():
    jev = FakeSystemOneModel(
        answer_fn=_answering(choices={"0": _choice("NONE", 0.9)})
    )
    classifier = _classifier(jev)
    assert classifier.classify(TEST_CASE, _show_indicator=False) is None
    assert classifier.reason == "The decision model found no label that fits."
    assert classifier.error is None


def test_multi_label_keeps_labels_at_the_review_confidence_highest_first():
    jev = FakeSystemOneModel(
        answer_fn=_answering(nouls={"0#0": 0.65, "0#1": 0.9, "0#2": 0.3})
    )
    classifier = _classifier(
        jev,
        labels=[*LABELS, Label("Returns", "Questions about returns.")],
        multi_label=True,
        review_config=ReviewConfig(min_confidence=0.6),
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    _, questions = jev.calls[0]
    assert all(isinstance(q, NoulQuestion) for q in questions.values())
    assert questions["0#1"].true == "Questions about deliveries."
    assert [(c.label, c.confidence) for c in classifier.classifications] == [
        ("Shipping", 0.9),
        ("Billing", 0.65),
    ]
    assert classifier.classifications[0].reason == (
        'Decided by fake-jev. Label "Shipping" (P=0.90, confidence=0.80).'
    )


def _discovering(first_confidence, second_confidence, proposals):
    jev = FakeSystemOneModel(
        answers=[
            SystemOneAnswers(
                choices={"0": _choice("Billing", first_confidence)}
            ),
            SystemOneAnswers(
                choices={"0": _choice("Parcel Tracking", second_confidence)}
            ),
        ]
    )
    proposer = ReplyModel({"labels": proposals})
    classifier = _classifier(
        jev,
        review_config=ReviewConfig(min_confidence=0.8),
        label_discovery_config=LabelDiscoveryConfig(
            model=proposer, min_confidence=0.8
        ),
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    return classifier, jev, proposer


def test_unsure_label_falls_back_to_a_discovered_label():
    classifier, jev, proposer = _discovering(
        0.5,
        0.85,
        [
            {"name": "Parcel Tracking", "description": "Traces where..."},
            {"name": "billing", "description": "duplicate of a declared label"},
            {"name": "Parcel Tracking", "description": "repeat"},
            {"name": "", "description": "empty"},
        ],
    )
    assert '"Billing"' in proposer.prompts[0]
    _, second = jev.calls[1]
    assert list(second["0"].options) == ["Parcel Tracking", "NONE"]
    assert classifier.label == "Parcel Tracking"
    assert classifier.classifications[0].new_label_description == (
        "Traces where..."
    )
    assert classifier.steps.classification.classifications == []
    assert classifier.steps.suggestion.classifications[0].label == (
        "Parcel Tracking"
    )


def test_discovered_label_below_its_confidence_is_dropped():
    classifier, _, _ = _discovering(
        0.5, 0.7, [{"name": "Parcel Tracking", "description": "d"}]
    )
    assert classifier.label is None
    assert classifier.reason == "The decision model found no label that fits."


def test_confident_declared_label_skips_discovery():
    classifier, jev, proposer = _discovering(0.9, 0.9, [])
    assert classifier.label == "Billing"
    assert len(jev.calls) == 1 and proposer.prompts == []


def test_issue_discovery_asks_which_issue_with_a_no_issue_option():
    jev = FakeSystemOneModel(
        answers=[
            SystemOneAnswers(nouls={"0#0": NoulAnswer(probability=0.1)}),
            SystemOneAnswers(choices={"0": _choice("NO_ISSUE", 0.9)}),
        ]
    )
    classifier = IssueClassifier(
        name="issues",
        labels=[Label("Late Delivery", "The parcel arrived late.")],
        eval_mode="decision_model",
        decision_model=jev,
        async_mode=False,
        evidence_config=None,
        label_discovery_config=LabelDiscoveryConfig(
            model=ReplyModel(
                {"labels": [{"name": "Parcel Tracking", "description": "d"}]}
            ),
            min_confidence=0.8,
        ),
    )
    assert classifier.classify(TEST_CASE, _show_indicator=False) is None
    _, second = jev.calls[1]
    question = second["0"]
    assert question.instructions["question"].startswith(
        'Which issue does `test_case` show according to the "issues" classifier?'
    )
    assert list(question.options) == ["Parcel Tracking", "NO_ISSUE"]
    assert question.options["NO_ISSUE"].startswith(
        "The agent behaved correctly"
    )
    assert classifier.reason == "The decision model found no label that fits."


@pytest.mark.asyncio
async def test_classifiers_sharing_a_model_are_asked_in_one_request():
    jev = FakeSystemOneModel(
        answer_fn=_answering(
            choices={"0": _choice("Shipping", 0.9)}, nouls={"1#0": 0.7}
        ),
        cost=0.02,
    )
    topic = _classifier(jev)
    issue = _classifier(
        jev,
        labels=[Label("Late Delivery", "The parcel arrived late.")],
        multi_label=True,
    )
    await a_classify_with_decision_model([topic, issue], TEST_CASE)
    assert len(jev.calls) == 1
    assert topic.label == "Shipping" and issue.label == "Late Delivery"
    assert topic.evaluation_cost == pytest.approx(0.01)


def test_questions_over_the_request_budget_are_split():
    jev = FakeSystemOneModel(
        answer_fn=_answering(nouls={"0#0": 0.9, "0#1": 0.9})
    )
    jev.max_request_tokens = 2_400
    classifier = _classifier(jev, multi_label=True)
    classify_with_decision_model([classifier], TEST_CASE)
    assert [list(questions) for _, questions in jev.calls] == [["0#0"], ["0#1"]]


def test_classifiers_asked_together_must_share_a_model():
    with pytest.raises(ValueError):
        classify_with_decision_model(
            [
                _classifier(FakeSystemOneModel()),
                _classifier(FakeSystemOneModel()),
            ],
            TEST_CASE,
        )


def test_discovery_needs_a_model_and_evidence_needs_an_llm():
    with pytest.raises(ValueError):
        _classifier(
            FakeSystemOneModel(), label_discovery_config=LabelDiscoveryConfig()
        )
    with pytest.raises(ValueError):
        _classifier(FakeSystemOneModel(), evidence_config=EvidenceConfig())


def test_system_one_model_keyword_is_deprecated():
    jev = FakeSystemOneModel()
    with pytest.warns(DeprecationWarning, match="'decision_model'"):
        classifier = Classifier(
            name="topic",
            labels=LABELS,
            eval_mode="decision_model",
            system_one_model=jev,
        )
    assert classifier.decision_model is jev


def test_system_one_eval_mode_is_deprecated():
    with pytest.warns(DeprecationWarning, match="'decision_model'"):
        classifier = Classifier(
            name="topic",
            labels=LABELS,
            eval_mode="system_one",
            decision_model=FakeSystemOneModel(),
        )
    assert classifier.eval_mode is EvalMode.SYSTEM_ONE


def test_both_decision_model_keywords_raise():
    jev = FakeSystemOneModel()
    with pytest.raises(TypeError, match="not both"):
        RefusalClassifier(
            eval_mode="decision_model",
            decision_model=jev,
            system_one_model=jev,
        )
