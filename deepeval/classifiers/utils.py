from __future__ import annotations

import logging
import re
from enum import Enum
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel

from deepeval.classifiers.classifier.steps.truncate import (
    cap_field,
    select_spans,
    serialize,
)

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.metrics.utils.decision import (
    _a_system_one_call,
    _accrue,
    _jsonable,
    _record_confidence,
    _system_one_active,
    _system_one_call,
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


_META_NOUNS = (
    r"(?:trace|content|interaction|conversation|input|output|data|message|"
    r"payload|json|response|record|log)"
)
_META_VERBS = (
    r"(?:shows|contains|reports|says|indicates|reveals|describes|presents|"
    r"returns|is|reflects|demonstrates|depicts)"
)
_META_ADVERBS = (
    r"(?:only|just|merely|simply|essentially|primarily|mostly|mainly)"
)
_META_PREFIX_RE = re.compile(
    r"^\s*(?:"
    rf"the content of the {_META_NOUNS}\s+(?:{_META_VERBS})\s+(?:that\s+)?"
    rf"|the {_META_NOUNS}(?:\s+{_META_ADVERBS})?\s+{_META_VERBS}\s+"
    r"(?:that\s+|a\s+|an\s+|only\s+|just\s+)?"
    rf"|the only (?:visible\s+|actual\s+|real\s+)?(?:{_META_NOUNS})\s+"
    rf"(?:is|shows|contains|{_META_VERBS})\s+"
    rf"|according to the {_META_NOUNS}[,:]?\s*"
    rf"|based on the {_META_NOUNS}[,:]?\s*"
    r"|looking at the " + _META_NOUNS + r"[,:]?\s*"
    rf"|from the {_META_NOUNS}[,:]?\s*"
    r")",
    re.IGNORECASE,
)
_MIN_CLEANED_LENGTH = 10


def clean_reason(reason: str) -> str:
    if not reason:
        return reason
    original = reason.strip()
    cleaned = _META_PREFIX_RE.sub("", original).strip()
    if cleaned == original or len(cleaned) < _MIN_CLEANED_LENGTH:
        return original
    return cleaned[:1].upper() + cleaned[1:]


_ABSENCE_LABEL_NAMES = {
    "none",
    "n/a",
    "na",
    "nothing",
    "other",
    "others",
    "unknown",
    "unclassified",
    "uncategorized",
    "not applicable",
    "miscellaneous",
}
_ABSENCE_LABEL_RE = re.compile(
    r"^(?:no|none|nothing|without)\b.*\b(?:issues?|problems?|errors?|failures?|"
    r"concerns?|complaints?|sentiments?|topics?|use cases?|intents?|labels?|"
    r"categor(?:y|ies)|match(?:es)?|applicable|relevant|clear|specific|found)\b",
    re.IGNORECASE,
)


def resolve_suggested_label(
    raw: Optional[str],
    description: Optional[str],
    reason: Optional[str],
    labels: List[Label],
    candidate_labels: List[Label],
    allow_none: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Map a judge answer that may name a new label onto ``(label, description)``.

    Existing labels win, then candidates (returned with their description),
    then a new label with the judge's description. A name that only says
    nothing applies is treated as ``NONE``.
    """
    existing = resolve_label(raw, labels, allow_none)
    if existing is not None or raw is None:
        return existing, None
    name = str(raw).strip()
    if not name or name.upper() == NONE_LABEL or is_absence_label(name):
        return (NONE_LABEL, None) if allow_none else (None, None)
    candidate = next(
        (c for c in candidate_labels if c.name.lower() == name.lower()), None
    )
    if candidate is not None:
        return candidate.name, candidate.description
    if description and description.strip():
        return name, description.strip()
    detail = (reason or "").strip().rstrip(
        "."
    ) or f'the evidence shows "{name}"'
    return (
        name,
        f"Cases where {detail[:1].lower() + detail[1:]}. "
        "Does not apply when supporting evidence is absent.",
    )


def is_absence_label(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in _ABSENCE_LABEL_NAMES or bool(
        _ABSENCE_LABEL_RE.match(normalized)
    )


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
SINGLE_TURN_CONTENT_PARAMS = {
    SingleTurnParams.INPUT: "Input",
    SingleTurnParams.ACTUAL_OUTPUT: "Actual Output",
    SingleTurnParams.EXPECTED_OUTPUT: "Expected Output",
    SingleTurnParams.CONTEXT: "Context",
    SingleTurnParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    SingleTurnParams.TOOLS_CALLED: "Tools Called",
    SingleTurnParams.EXPECTED_TOOLS: "Expected Tools",
}

MULTI_TURN_CONTENT_PARAMS = {
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
    for param, display in SINGLE_TURN_CONTENT_PARAMS.items():
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


_TRACE_FIELDS = {
    "name": "name",
    "input": "input",
    "actual_output": "output",
    "expected_output": "expectedOutput",
    "retrieval_context": "retrievalContext",
    "context": "context",
    "tools_called": "toolsCalled",
    "expected_tools": "expectedTools",
}
_CAPPED_SPAN_FIELDS = {
    "error": "error",
    "input": "input",
    "output": "output",
    "tools_called": "toolsCalled",
}


def construct_trace_content(content: Union[Dict[str, Any], str]) -> str:
    return content if isinstance(content, str) else serialize(content)


def construct_trace_dict(test_case: LLMTestCase) -> Dict[str, Any]:
    trace = {
        key: _jsonable_field(getattr(test_case, attr, None))
        for attr, key in _TRACE_FIELDS.items()
        if _is_populated(getattr(test_case, attr, None))
    }
    spans = select_spans(_flatten_spans(test_case._trace_dict, []))
    if spans:
        trace["spans"] = spans
    return trace


def _flatten_spans(span: Optional[Dict], spans: List[Dict]) -> List[Dict]:
    if not isinstance(span, dict):
        return spans
    entry: Dict[str, Any] = {
        "name": span.get("name") or "",
        "type": _jsonable_field(span.get("type")),
        "status": "ERRORED" if span.get("error") else "SUCCESS",
    }
    for attr, key in _CAPPED_SPAN_FIELDS.items():
        capped = cap_field(_jsonable_field(span.get(attr)))
        if capped is not None:
            entry[key] = capped
    spans.append(entry)
    for child in span.get("children") or []:
        _flatten_spans(child, spans)
    return spans


def _jsonable_field(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable_field(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_field(v) for k, v in value.items()}
    return _jsonable(value)


def construct_multi_turn_content(test_case: ConversationalTestCase) -> str:
    text = ""
    for param, display in MULTI_TURN_CONTENT_PARAMS.items():
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
# Under the `system_one` eval mode the label is one Jev Choice over the
# labels, the reason is deterministic text and Jev errors surface (no
# fallback). Otherwise it is one schema-constrained LLM call. Classifiers have
# no `hybrid`: it resolves to `llm` (see ``resolve_classifier_eval_mode``).
# Mirrors ``deepeval.metrics.utils.decision`` for metrics and reuses its
# plumbing; see EXPERIMENTAL.md.


@dataclass
class SystemOneClassifySpec:
    """One Choice question over a classifier's labels.

    ``options`` maps each label name to its description (or ``None``), the
    same boundary the LLM prompt shows. When ``include_reason`` is set the
    reason is built from the answer itself; otherwise it is ``None``.
    """

    instructions: Any
    options: Dict[str, Any]
    state: Dict[str, Any]
    include_reason: bool


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


def _classification_accept(
    classifier: BaseClassifier,
    spec: SystemOneClassifySpec,
    answers: Any,
    cost: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Book the Jev answer and return ``(label, reason)``."""
    from deepeval.metrics.utils.system_one import format_classification_reason

    _accrue(classifier, cost)
    _record_confidence(classifier, answers)
    answer = answers["label"]
    if not spec.include_reason:
        return answer.choice, None
    return answer.choice, format_classification_reason(classifier, answer)


def generate_classification(
    classifier: BaseClassifier,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    system_one: Optional[SystemOneClassifySpec] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask the judge for ``(label, reason)`` from a closed label set.

    Under the ``system_one`` eval mode the label is one Choice over
    ``system_one.options`` and the reason is deterministic text. Otherwise
    the whole thing is one schema-constrained LLM call against
    ``schema_cls`` (``label``, ``reason``).
    """
    if _system_one_active(classifier, system_one):
        answers, cost = _system_one_call(
            classifier,
            classifier.system_one_model.choice,
            _jsonable(system_one.state),
            _classify_question(system_one),
        )
        return _classification_accept(classifier, system_one, answers, cost)

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
        answers, cost = await _a_system_one_call(
            classifier,
            classifier.system_one_model.a_choice,
            _jsonable(system_one.state),
            _classify_question(system_one),
        )
        return _classification_accept(classifier, system_one, answers, cost)

    return await a_generate_with_schema_and_extract(
        metric=classifier,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda r: (r.label, r.reason),
        extract_json=lambda d: (d.get("label"), d.get("reason")),
    )


###############################################
# Test case -> System One state
###############################################
#
# JSON counterparts of the string builders above, for the state handed to a
# System One model when the eval mode uses one. Same fields, same
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
        test_case, SINGLE_TURN_CONTENT_PARAMS
    )
    if not fields:
        raise ValueError(
            "Cannot classify an LLMTestCase with no populated fields (input, actual_output, expected_output, context, retrieval_context, tools_called, expected_tools)."
        )
    return {"test_case": fields}


def construct_multi_turn_state(test_case: ConversationalTestCase) -> Dict:
    state: Dict = {"turns": construct_turns(test_case)}
    fields = _experimental_system_one_fields(
        test_case, MULTI_TURN_CONTENT_PARAMS
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
