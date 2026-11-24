import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel

from deepeval.classifiers.base_classifier import NONE_LABEL, Label
from deepeval.classifiers.classifier.steps.classification import (
    PROVIDER_ERRORS,
    ClassificationStep,
)
from deepeval.classifiers.classifier.steps.trace import construct_trace_dict
from deepeval.classifiers.classifier.steps.truncate import (
    fit_text,
    fit_trace,
    fit_turns,
)
from deepeval.classifiers.utils import (
    construct_multi_turn_state,
    construct_single_turn_state,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils.decision import (
    _accrue,
    _jsonable,
    _record_confidence,
)
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.metrics.utils.models import initialize_model
from deepeval.metrics.utils.system_one import format_classification_reason
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.limits import context_limits, estimate_tokens
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulQuestion,
    SystemOneAnswers,
    SystemOneQuestion,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.test_run.api import Classification

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import Classifier

CANDIDATE_LABEL_COUNT = 5
NO_FITTING_LABEL_REASON = "The decision model found no label that fits."
NONE_OPTION_DESCRIPTION = (
    "None of the other options fit. Pick this when no label's criteria match."
)
NO_ISSUE_LABEL = "NO_ISSUE"
NO_ISSUE_OPTION_DESCRIPTION = (
    "The agent behaved correctly; none of these describe a problem in what the "
    "agent did."
)
DEFAULT_LABEL_MIN_PROBABILITY = 0.5
REQUEST_TOKEN_MARGIN = 2_000
TEMPLATE_CLASS = "Classifier"

TestCase = Union[LLMTestCase, ConversationalTestCase]


class CandidateLabel(BaseModel):
    name: str
    description: str


class CandidateLabels(BaseModel):
    labels: List[CandidateLabel]


@dataclass(frozen=True)
class Shape:
    multi_turn: bool
    traced: bool
    metadata_description: Optional[str] = None


@dataclass(frozen=True)
class Ask:

    classifier: "Classifier"
    labels: List[Label]
    per_label: bool
    min_confidence: Optional[float]
    discovered: bool = False


def classify_with_decision_model(
    classifiers: Sequence["Classifier"],
    test_case: TestCase,
    metadata_description: Optional[str] = None,
) -> None:
    state, shape = _start(classifiers, test_case, metadata_description)
    started = time.perf_counter()
    first = _first_asks(classifiers)
    if first:
        _accept(
            first, _decide(classifiers[0].decision_model, state, first, shape)
        )
    _record_classification(classifiers, time.perf_counter() - started)
    started = time.perf_counter()
    second = [
        ask
        for ask in (
            _discovery_ask(
                classifier, _propose_labels(classifier, state, shape)
            )
            for classifier in _undecided(classifiers)
        )
        if ask is not None
    ]
    if second:
        _accept(
            second,
            _decide(classifiers[0].decision_model, state, second, shape),
        )
    _finish(classifiers, time.perf_counter() - started)


async def a_classify_with_decision_model(
    classifiers: Sequence["Classifier"],
    test_case: TestCase,
    metadata_description: Optional[str] = None,
) -> None:
    state, shape = _start(classifiers, test_case, metadata_description)
    started = time.perf_counter()
    first = _first_asks(classifiers)
    if first:
        _accept(
            first,
            await _a_decide(classifiers[0].decision_model, state, first, shape),
        )
    _record_classification(classifiers, time.perf_counter() - started)
    started = time.perf_counter()
    undecided = _undecided(classifiers)
    proposals = await asyncio.gather(
        *(
            _a_propose_labels(classifier, state, shape)
            for classifier in undecided
        )
    )
    second = [
        ask
        for ask in (
            _discovery_ask(classifier, candidates)
            for classifier, candidates in zip(undecided, proposals)
        )
        if ask is not None
    ]
    if second:
        _accept(
            second,
            await _a_decide(
                classifiers[0].decision_model, state, second, shape
            ),
        )
    _finish(classifiers, time.perf_counter() - started)


###############################################
# Rounds
###############################################


def _start(
    classifiers: Sequence["Classifier"],
    test_case: TestCase,
    metadata_description: Optional[str],
):
    if not classifiers:
        raise ValueError(
            "classify_with_decision_model needs at least one classifier."
        )
    model = classifiers[0].decision_model
    for classifier in classifiers:
        if not classifier.eval_mode.uses_system_one:
            raise ValueError(
                f"Classifier '{classifier.name}' is not in the decision_model eval mode."
            )
        if classifier.decision_model is not model:
            raise ValueError(
                "Classifiers asked together must share one decision_model."
            )
        classifier._check_test_case(test_case)
        classifier._reset_usage()
    return _state(classifiers[0], test_case, metadata_description)


def _first_asks(classifiers: Sequence["Classifier"]) -> List[Ask]:
    return [
        Ask(
            classifier,
            classifier.labels,
            classifier.multi_label,
            (
                classifier.review_config.min_confidence
                if classifier.review_config
                else None
            ),
        )
        for classifier in classifiers
        if classifier.labels
    ]


def _undecided(classifiers: Sequence["Classifier"]) -> List["Classifier"]:
    return [
        classifier
        for classifier in classifiers
        if classifier.label_discovery_config and not classifier.classifications
    ]


def _discovery_ask(
    classifier: "Classifier", candidates: List[Label]
) -> Optional[Ask]:
    if not candidates:
        return None
    return Ask(
        classifier,
        candidates,
        False,
        classifier.label_discovery_config.min_confidence,
        discovered=True,
    )


def _accept(asks: List[Ask], answers: SystemOneAnswers) -> None:
    for index, ask in enumerate(asks):
        if ask.per_label:
            answered = {
                key: answers.nouls[key]
                for key in _noul_keys(index, ask)
                if key in answers.nouls
            }
        else:
            key = str(index)
            answered = (
                {key: answers.choices[key]} if key in answers.choices else {}
            )
        _record_confidence(ask.classifier, answered)
        ask.classifier.classifications = (
            _noul_classifications(index, ask, answered)
            if ask.per_label
            else _choice_classifications(index, ask, answered)
        )


def _choice_classifications(
    index: int, ask: Ask, answered: Dict[str, Any]
) -> List[Classification]:
    answer = answered.get(str(index))
    labels = {label.name: label for label in ask.labels}
    if answer is None or answer.choice not in labels:
        return []
    confidence = round(float(answer.confidence), 4)
    if ask.min_confidence is not None and confidence < ask.min_confidence:
        return []
    return [_classification(ask, labels[answer.choice], confidence, answer)]


def _noul_classifications(
    index: int, ask: Ask, answered: Dict[str, Any]
) -> List[Classification]:
    min_probability = (
        DEFAULT_LABEL_MIN_PROBABILITY
        if ask.min_confidence is None
        else ask.min_confidence
    )
    kept = [
        (label, round(float(answered[key].probability), 4), answered[key])
        for label, key in zip(ask.labels, _noul_keys(index, ask))
        if key in answered and answered[key].probability >= min_probability
    ]
    kept.sort(key=lambda entry: entry[1], reverse=True)
    return [
        _classification(
            ask,
            label,
            confidence,
            ChoiceAnswer(
                choice=label.name,
                probabilities={label.name: answer.probability},
                confidence=answer.confidence,
            ),
        )
        for label, confidence, answer in kept
    ]


def _classification(
    ask: Ask, label: Label, confidence: float, answer: ChoiceAnswer
) -> Classification:
    classifier = ask.classifier
    return Classification(
        name=classifier.name,
        label=label.name,
        reason=(
            format_classification_reason(classifier, answer)
            if classifier.include_reason
            else None
        ),
        new_label_description=label.description if ask.discovered else None,
        confidence=confidence,
    )


def _record_classification(
    classifiers: Sequence["Classifier"], seconds: float
) -> None:
    for classifier in classifiers:
        classifier.steps.classification = ClassificationStep(
            classifications=[
                classification.model_copy(deep=True)
                for classification in classifier.classifications
            ],
            seconds=seconds,
        )


def _finish(
    classifiers: Sequence["Classifier"], discovery_seconds: float
) -> None:
    for classifier in classifiers:
        if (
            classifier.label_discovery_config
            and not classifier.steps.classification.classifications
        ):
            classifier.steps.suggestion = ClassificationStep(
                classifications=[
                    classification.model_copy(deep=True)
                    for classification in classifier.classifications
                ],
                seconds=discovery_seconds,
            )
        top = (
            classifier.classifications[0]
            if classifier.classifications
            else None
        )
        classifier.label = top.label if top else None
        classifier.reason = (
            (top.reason if top else NO_FITTING_LABEL_REASON)
            if classifier.include_reason
            else None
        )
        if top is None and not classifier.allow_none:
            classifier.error = "Judge returned no usable label."


###############################################
# Requests
###############################################


def _decide(
    model: DeepEvalBaseSystemOneModel,
    state: Dict[str, Any],
    asks: List[Ask],
    shape: Shape,
) -> SystemOneAnswers:
    results = [
        model.decide(state, chunk)
        for chunk in _chunks(model, state, _questions(asks, shape))
    ]
    return _merged(asks, results)


async def _a_decide(
    model: DeepEvalBaseSystemOneModel,
    state: Dict[str, Any],
    asks: List[Ask],
    shape: Shape,
) -> SystemOneAnswers:
    results = await asyncio.gather(
        *(
            model.a_decide(state, chunk)
            for chunk in _chunks(model, state, _questions(asks, shape))
        )
    )
    return _merged(asks, list(results))


def _merged(asks: List[Ask], results: List[Any]) -> SystemOneAnswers:
    merged = SystemOneAnswers()
    for answers, cost in results:
        merged.nouls.update(answers.nouls)
        merged.choices.update(answers.choices)
        share = (
            cost if cost is None or len(asks) == 1 else float(cost) / len(asks)
        )
        for ask in asks:
            _accrue(ask.classifier, share)
    return merged


def _chunks(
    model: DeepEvalBaseSystemOneModel,
    state: Dict[str, Any],
    questions: Dict[str, SystemOneQuestion],
) -> List[Dict[str, SystemOneQuestion]]:
    limit = context_limits(model)["max_request_tokens"]
    if limit is None:
        return [questions]
    budget = limit - estimate_tokens(state) - REQUEST_TOKEN_MARGIN
    chunks: List[Dict[str, SystemOneQuestion]] = [{}]
    used = 0
    for key, question in questions.items():
        tokens = estimate_tokens({key: question.model_dump()})
        if chunks[-1] and used + tokens > budget:
            chunks.append({})
            used = 0
        chunks[-1][key] = question
        used += tokens
    return chunks


###############################################
# Questions
###############################################


def _questions(asks: List[Ask], shape: Shape) -> Dict[str, SystemOneQuestion]:
    questions: Dict[str, SystemOneQuestion] = {}
    for index, ask in enumerate(asks):
        if ask.per_label:
            questions.update(_noul_questions(index, ask, shape))
        else:
            questions[str(index)] = _choice_question(ask, shape)
    return questions


def _choice_question(ask: Ask, shape: Shape) -> ChoiceQuestion:
    classifier = ask.classifier
    options: Dict[str, Optional[str]] = {
        label.name: label.description for label in ask.labels
    }
    if classifier.finds_issues:
        options[NO_ISSUE_LABEL] = NO_ISSUE_OPTION_DESCRIPTION
    elif classifier.allow_none:
        options[NONE_LABEL] = NONE_OPTION_DESCRIPTION
    return ChoiceQuestion(
        instructions=_instructions(
            classifier,
            classifier._get_prompt(
                "_experimental_decision_model_classify",
                template_class=TEMPLATE_CLASS,
                name=classifier.name,
                allow_none=classifier.allow_none,
                issue=classifier.finds_issues,
                multi_turn=shape.multi_turn,
                traced=shape.traced,
            ),
            shape,
        ),
        options=options,
    )


def _noul_questions(
    index: int, ask: Ask, shape: Shape
) -> Dict[str, NoulQuestion]:
    classifier = ask.classifier
    return {
        key: NoulQuestion(
            instructions=_instructions(
                classifier,
                classifier._get_prompt(
                    "_experimental_decision_model_label_applies",
                    template_class=TEMPLATE_CLASS,
                    name=classifier.name,
                    label=label.name,
                    multi_turn=shape.multi_turn,
                    traced=shape.traced,
                ),
                shape,
            ),
            true=label.description or f'The content shows "{label.name}".',
            false="The content does not meet the label's description.",
        )
        for label, key in zip(ask.labels, _noul_keys(index, ask))
    }


def _noul_keys(index: int, ask: Ask) -> List[str]:
    return [f"{index}#{label_index}" for label_index in range(len(ask.labels))]


def _instructions(
    classifier: "Classifier", question: str, shape: Shape
) -> Dict[str, str]:
    instructions = {
        "question": question,
        "content_format": _content_format(classifier, shape),
    }
    if classifier.description:
        instructions["classifier_purpose"] = classifier.description
    app_description = _app_description(classifier)
    if app_description:
        instructions["application"] = app_description
    return instructions


def _content_format(classifier: "Classifier", shape: Shape) -> str:
    return classifier._get_prompt(
        "_experimental_decision_model_content_format",
        template_class=TEMPLATE_CLASS,
        multi_turn=shape.multi_turn,
        traced=shape.traced,
        metadata_description=shape.metadata_description,
    )


def _app_description(classifier: "Classifier") -> Optional[str]:
    profile = classifier.agent_profile
    return profile.description if profile else None


###############################################
# State
###############################################


def _state(
    classifier: "Classifier",
    test_case: TestCase,
    metadata_description: Optional[str],
):
    state, multi_turn, traced = _content_state(classifier, test_case)
    with_metadata = bool(metadata_description and test_case.metadata)
    if with_metadata:
        state["metadata"] = test_case.metadata
    shape = Shape(
        multi_turn=multi_turn,
        traced=traced,
        metadata_description=metadata_description if with_metadata else None,
    )
    return _jsonable(state), shape


def _content_state(classifier: "Classifier", test_case: TestCase):
    max_chars = classifier.max_content_chars
    if isinstance(test_case, ConversationalTestCase):
        state = construct_multi_turn_state(test_case)
        state["turns"] = fit_turns(state["turns"], max_chars)
        return state, True, False
    if test_case._trace_dict is not None:
        return (
            {"trace": fit_trace(construct_trace_dict(test_case), max_chars)},
            False,
            True,
        )
    state = construct_single_turn_state(test_case)
    state["test_case"] = {
        field: fit_text(value, max_chars) if isinstance(value, str) else value
        for field, value in state["test_case"].items()
    }
    return state, False, False


###############################################
# Label discovery
###############################################


def _propose_labels(
    classifier: "Classifier", state: Dict[str, Any], shape: Shape
) -> List[Label]:
    try:
        proposed = generate_with_schema_and_extract(
            metric=classifier,
            prompt=_proposal_prompt(classifier, state, shape),
            schema_cls=CandidateLabels,
            extract_schema=lambda labels: labels,
            extract_json=CandidateLabels.model_validate,
            model=_discovery_model(classifier),
        )
    except (*PROVIDER_ERRORS, ValueError, DeepEvalError):
        classifier.error = None
        return []
    return _candidates(classifier, proposed)


async def _a_propose_labels(
    classifier: "Classifier", state: Dict[str, Any], shape: Shape
) -> List[Label]:
    try:
        proposed = await a_generate_with_schema_and_extract(
            metric=classifier,
            prompt=_proposal_prompt(classifier, state, shape),
            schema_cls=CandidateLabels,
            extract_schema=lambda labels: labels,
            extract_json=CandidateLabels.model_validate,
            model=_discovery_model(classifier),
        )
    except (*PROVIDER_ERRORS, ValueError, DeepEvalError):
        classifier.error = None
        return []
    return _candidates(classifier, proposed)


def _discovery_model(classifier: "Classifier"):
    model, _ = initialize_model(classifier.label_discovery_config.model)
    return model


def _proposal_prompt(
    classifier: "Classifier", state: Dict[str, Any], shape: Shape
) -> str:
    return classifier._get_prompt(
        "_experimental_decision_model_candidate_labels",
        template_class=TEMPLATE_CLASS,
        content_format=_content_format(classifier, shape),
        app_description=_app_description(classifier),
        name=classifier.name,
        description=classifier.description,
        existing_labels=", ".join(
            f'"{label.name}"' for label in classifier.labels
        ),
        count=CANDIDATE_LABEL_COUNT,
        content=json.dumps(state, ensure_ascii=False, default=str),
    )


def _candidates(
    classifier: "Classifier", proposed: CandidateLabels
) -> List[Label]:
    taken = {label.name.lower() for label in classifier.labels}
    candidates: List[Label] = []
    for candidate in proposed.labels:
        name, description = (
            candidate.name.strip(),
            candidate.description.strip(),
        )
        if not name or not description or name.lower() in taken:
            continue
        taken.add(name.lower())
        candidates.append(Label(name=name, description=description))
    return candidates[:CANDIDATE_LABEL_COUNT]
