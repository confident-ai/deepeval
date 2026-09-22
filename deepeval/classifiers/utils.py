from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.metrics.utils.decision import (
    _accrue,
    _jsonable,
    _system_one_active,
)
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.metrics.utils.turns import convert_turn_to_dict
from deepeval.models.system_one.schema import ChoiceQuestion
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MultiTurnParams,
    SingleTurnParams,
    ToolCall,
)

logger = logging.getLogger(__name__)


###############################################
# Labels
###############################################


def normalize_labels(labels: Sequence[Union[str, Label]]) -> List[Label]:
    if not labels:
        raise ValueError("A classifier needs at least one label.")

    normalized: List[Label] = []
    for label in labels:
        if isinstance(label, Label):
            normalized.append(label)
        elif isinstance(label, str):
            normalized.append(Label(name=label))
        else:
            raise TypeError(
                f"Labels must be 'str' or 'Label', got {type(label).__name__}."
            )

    for label in normalized:
        if not label.name or not label.name.strip():
            raise ValueError("Label names cannot be empty.")
        if label.name.strip().upper() == NONE_LABEL:
            # Internal sentinel the judge uses to decline classifying
            # (see `allow_none`); a user label with the same name would be
            # indistinguishable from it.
            raise ValueError(
                f"'{label.name}' cannot be used as a label name. If you want a "
                "'no classification' outcome, set allow_none=True instead."
            )

    seen = Counter(label.name.strip().lower() for label in normalized)
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate labels (case-insensitive): {', '.join(duplicates)}."
        )

    return normalized


def resolve_label(
    raw: Optional[str], labels: List[Label], allow_none: bool = False
) -> Optional[str]:
    """Map the judge's raw answer onto a declared label name.

    Returns ``NONE_LABEL`` when the judge declined to classify and
    ``allow_none`` permits that, and ``None`` when the answer matches nothing,
    so the caller can record an error instead of silently inventing a label.
    """
    if raw is None:
        return None
    candidate = str(raw).strip()
    if candidate.upper() == NONE_LABEL:
        return NONE_LABEL if allow_none else None
    for label in labels:
        if label.name == candidate:
            return label.name
    lowered = candidate.lower()
    for label in labels:
        if label.name.lower() == lowered:
            return label.name
    return None


def format_labels(labels: List[Label]) -> str:
    lines = []
    for label in labels:
        if label.description:
            lines.append(f"- {label.name}: {label.description}")
        else:
            lines.append(f"- {label.name}")
    return "\n".join(lines)


###############################################
# Test case -> prompt content
###############################################

# Ordered (param -> display name). Every populated field is shown to the judge.
_SINGLE_TURN_CONTENT_PARAMS = {
    SingleTurnParams.INPUT: "Input",
    SingleTurnParams.ACTUAL_OUTPUT: "Actual Output",
    SingleTurnParams.EXPECTED_OUTPUT: "Expected Output",
    SingleTurnParams.CONTEXT: "Context",
    SingleTurnParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    SingleTurnParams.TOOLS_CALLED: "Tools Called",
    SingleTurnParams.EXPECTED_TOOLS: "Expected Tools",
}

_MULTI_TURN_CONTENT_PARAMS = {
    MultiTurnParams.SCENARIO: "Scenario",
    MultiTurnParams.EXPECTED_OUTCOME: "Expected Outcome",
    MultiTurnParams.USER_DESCRIPTION: "User Description",
    MultiTurnParams.CONTEXT: "Context",
    MultiTurnParams.CHATBOT_ROLE: "Chatbot Role",
}

_TURN_PARAMS = [
    MultiTurnParams.ROLE,
    MultiTurnParams.CONTENT,
    MultiTurnParams.RETRIEVAL_CONTEXT,
    MultiTurnParams.TOOLS_CALLED,
]


def _is_populated(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    return True


def construct_single_turn_content(test_case: LLMTestCase) -> str:
    text = ""
    for param, display in _SINGLE_TURN_CONTENT_PARAMS.items():
        value = getattr(test_case, param.value, None)
        if not _is_populated(value):
            continue
        if isinstance(value, list) and value and isinstance(value[0], ToolCall):
            value = [repr(tool) for tool in value]
        text += f"{display}:\n{value}\n\n"
    if not text:
        raise ValueError(
            "Cannot classify an LLMTestCase with no populated fields (input, actual_output, expected_output, context, retrieval_context, tools_called, expected_tools)."
        )
    return text.rstrip()


def construct_multi_turn_content(test_case: ConversationalTestCase) -> str:
    text = ""
    for param, display in _MULTI_TURN_CONTENT_PARAMS.items():
        value = getattr(test_case, param.value, None)
        if not _is_populated(value):
            continue
        text += f"{display}:\n{value}\n\n"
    return text.rstrip()


def construct_turns(test_case: ConversationalTestCase) -> List[Dict]:
    if not test_case.turns:
        raise ValueError(
            "Cannot classify a ConversationalTestCase with no turns."
        )
    return [
        convert_turn_to_dict(turn, _TURN_PARAMS) for turn in test_case.turns
    ]


###############################################
# Classification decision
###############################################
#
# The single place where a classifier turns a prompt into ``(label, reason)``.
# Under DEEPEVAL_MODE=experimental the label is one System One (Jev) Choice
# over the labels and the LLM only writes the reason; otherwise it is one
# schema-constrained LLM call. Mirrors ``deepeval.metrics.utils.decision`` for
# metrics and reuses its plumbing; see EXPERIMENTAL.md.


@dataclass
class SystemOneClassifySpec:
    """One Choice question over a classifier's labels.

    ``options`` maps each label name to its description (or ``None``), the
    same boundary the LLM prompt shows. ``reason_prompt`` receives the chosen
    label, the per-label probabilities and Jev's confidence; when it is
    ``None`` no LLM call is made at all and the reason is ``None``.
    """

    instructions: Any
    options: Dict[str, Any]
    state: Dict[str, Any]
    reason_prompt: Optional[Callable[[str, Dict[str, float], float], Any]]
    reason_schema_cls: Type[Any]


def _classify_question(
    spec: SystemOneClassifySpec,
) -> Dict[str, ChoiceQuestion]:
    return {
        "label": ChoiceQuestion(
            instructions=spec.instructions,
            options={
                name: _jsonable(desc) for name, desc in spec.options.items()
            },
        )
    }


def generate_classification(
    classifier: BaseClassifier,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    system_one: Optional[SystemOneClassifySpec] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask the judge for ``(label, reason)`` from a closed label set.

    Under ``DEEPEVAL_MODE=experimental`` the label is one System One Choice
    over ``system_one.options``; the LLM only writes the reason (when
    ``system_one.reason_prompt`` is set). Otherwise the whole thing is one
    schema-constrained LLM call against ``schema_cls`` (``label``, ``reason``).
    """
    if _system_one_active(classifier, system_one):
        answers, cost = classifier.system_one_model.choice(
            _jsonable(system_one.state), _classify_question(system_one)
        )
        _accrue(classifier, cost)
        answer = answers["label"]
        if system_one.reason_prompt is None:
            return answer.choice, None
        reason = generate_with_schema_and_extract(
            metric=classifier,
            prompt=system_one.reason_prompt(
                answer.choice, answer.probabilities, answer.confidence
            ),
            schema_cls=system_one.reason_schema_cls,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )
        return answer.choice, reason

    return generate_with_schema_and_extract(
        metric=classifier,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda r: (r.label, r.reason),
        extract_json=lambda d: (d.get("label"), d.get("reason")),
    )


async def a_generate_classification(
    classifier: BaseClassifier,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    system_one: Optional[SystemOneClassifySpec] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Async counterpart of ``generate_classification``."""
    if _system_one_active(classifier, system_one):
        answers, cost = await classifier.system_one_model.a_choice(
            _jsonable(system_one.state), _classify_question(system_one)
        )
        _accrue(classifier, cost)
        answer = answers["label"]
        if system_one.reason_prompt is None:
            return answer.choice, None
        reason = await a_generate_with_schema_and_extract(
            metric=classifier,
            prompt=system_one.reason_prompt(
                answer.choice, answer.probabilities, answer.confidence
            ),
            schema_cls=system_one.reason_schema_cls,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )
        return answer.choice, reason

    return await a_generate_with_schema_and_extract(
        metric=classifier,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda r: (r.label, r.reason),
        extract_json=lambda d: (d.get("label"), d.get("reason")),
    )


###############################################
# Test case -> System One state (experimental)
###############################################
#
# JSON counterparts of the string builders above, for the state handed to a
# System One model under DEEPEVAL_MODE=experimental. Same fields, same
# populated-only rule; see EXPERIMENTAL.md.


def _experimental_system_one_fields(test_case, params) -> Dict:
    fields: Dict = {}
    for param in params:
        value = getattr(test_case, param.value, None)
        if not _is_populated(value):
            continue
        if isinstance(value, list) and value and isinstance(value[0], ToolCall):
            value = [repr(tool) for tool in value]
        fields[param.value] = value
    return fields


def construct_single_turn_state(test_case: LLMTestCase) -> Dict:
    fields = _experimental_system_one_fields(
        test_case, _SINGLE_TURN_CONTENT_PARAMS
    )
    if not fields:
        raise ValueError(
            "Cannot classify an LLMTestCase with no populated fields (input, actual_output, expected_output, context, retrieval_context, tools_called, expected_tools)."
        )
    return {"test_case": fields}


def construct_multi_turn_state(test_case: ConversationalTestCase) -> Dict:
    state: Dict = {"turns": construct_turns(test_case)}
    fields = _experimental_system_one_fields(
        test_case, _MULTI_TURN_CONTENT_PARAMS
    )
    if fields:
        state["test_case"] = fields
    return state


###############################################
# Copying
###############################################


def copy_classifiers(
    classifiers: Sequence[BaseClassifier],
) -> List[BaseClassifier]:
    """Classifier counterpart of ``copy_metrics``: one fresh instance per
    classifier so concurrently evaluated test cases never share state."""
    return [classifier.copy() for classifier in classifiers]


###############################################
# Run-level validation
###############################################


def validate_classifiers(
    test_cases: Optional[Sequence[Union[LLMTestCase, ConversationalTestCase]]],
    classifiers: Optional[Sequence[BaseClassifier]],
) -> None:
    """Validate a run's classifiers against its test cases before any judge call.

    - every classifier is a ``BaseClassifier`` (error)
    - classifier names are unique within the run (error)
    - an ``expected_labels`` value that is not one of the classifier's labels (error)
    - an ``expected_labels`` key that matches no classifier in the run (one
      aggregated warning, so datasets can carry expectations for classifiers
      not part of this run)
    """
    if not classifiers:
        return

    for classifier in classifiers:
        if not isinstance(classifier, BaseClassifier):
            raise ValueError(
                f"All 'classifiers' must be instances of 'BaseClassifier', got {type(classifier).__name__}."
            )

    names = Counter(classifier.name for classifier in classifiers)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Classifier names must be unique within a run, duplicated: {', '.join(duplicates)}."
        )

    by_name: Dict[str, BaseClassifier] = {c.name: c for c in classifiers}
    unknown_keys: Counter = Counter()

    for test_case in test_cases or []:
        expected_labels = getattr(test_case, "expected_labels", None)
        if not expected_labels:
            continue
        for key, value in expected_labels.items():
            classifier = by_name.get(key)
            if classifier is None:
                unknown_keys[key] += 1
                continue
            if value not in classifier.labels:
                raise ValueError(
                    f"expected_labels['{key}'] = '{value}' is not one of classifier '{key}' labels: {', '.join(str(l) for l in classifier.labels)}."
                )

    if unknown_keys:
        details = ", ".join(
            f"'{key}' ({count} test case{'s' if count != 1 else ''})"
            for key, count in sorted(unknown_keys.items())
        )
        logger.warning(
            f"expected_labels keys matched no classifier in this run: {details}. "
            "These expectations were ignored."
        )
