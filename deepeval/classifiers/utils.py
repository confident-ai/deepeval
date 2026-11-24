from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
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
        raise ValueError(
            "A classifier needs at least one label, or a label_discovery_config to discover them."
        )

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
    original = str(reason).strip()
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
    "none found",
}


class LabelOrigin(str, Enum):
    DECLARED = "declared"
    CANDIDATE = "candidate"
    NEW = "new"
    NONE = "none"


@dataclass(frozen=True)
class ResolvedLabel:
    """A judge answer mapped onto a label.

    ``name`` is ``None`` when the answer matched nothing (an error for the
    caller to record). ``description`` is set only for candidate and new
    labels, since declared labels already carry their own.
    """

    name: Optional[str]
    origin: Optional[LabelOrigin] = None
    description: Optional[str] = None

    @property
    def unresolved(self) -> bool:
        return self.name is None

    @property
    def is_none(self) -> bool:
        return self.origin is LabelOrigin.NONE


def resolve_suggested_label(
    raw: Optional[str],
    *,
    description: Optional[str],
    reason: Optional[str],
    labels: List[Label],
    candidate_labels: List[Label],
    allow_none: bool,
    classifier_name: str,
) -> ResolvedLabel:
    """Map a judge answer that may name a new label onto a ``ResolvedLabel``.

    Declared labels win, then candidates (with their description), then a new
    label with the judge's description. A name that only says nothing applies
    is treated as ``NONE``.
    """
    existing = resolve_label(raw, labels, allow_none)
    if existing == NONE_LABEL:
        return ResolvedLabel(NONE_LABEL, LabelOrigin.NONE)
    if existing is not None:
        return ResolvedLabel(existing, LabelOrigin.DECLARED)
    if raw is None:
        return ResolvedLabel(None)
    name = str(raw).strip()
    if (
        not name
        or name.upper() == NONE_LABEL
        or _is_absence_label(name, classifier_name)
    ):
        return (
            ResolvedLabel(NONE_LABEL, LabelOrigin.NONE)
            if allow_none
            else ResolvedLabel(None)
        )
    candidate = next(
        (c for c in candidate_labels if c.name.lower() == name.lower()), None
    )
    if candidate is not None:
        return ResolvedLabel(
            candidate.name, LabelOrigin.CANDIDATE, candidate.description
        )
    if description and description.strip():
        return ResolvedLabel(name, LabelOrigin.NEW, description.strip())
    return ResolvedLabel(
        name, LabelOrigin.NEW, _fallback_description(name, reason)
    )


def _fallback_description(name: str, reason: Optional[str]) -> str:
    detail = (reason or "").strip().rstrip(
        "."
    ) or f'the evidence shows "{name}"'
    # Lowercase the first letter to fit "Cases where ...", unless it starts an
    # acronym such as "API".
    if not detail[1:2].isupper():
        detail = detail[:1].lower() + detail[1:]
    return (
        f"Cases where {detail}. "
        "Does not apply when supporting evidence is absent."
    )


def _is_absence_label(name: str, classifier_name: str) -> bool:
    normalized = " ".join(name.lower().split())
    if normalized in _ABSENCE_LABEL_NAMES:
        return True
    singular = re.escape(" ".join(classifier_name.lower().split()).rstrip("s"))
    return bool(
        singular
        and re.fullmatch(
            rf"(?:no|none|without) (?:\w+ ){{0,2}}{singular}(?:s|es)?(?: \w+)?",
            normalized,
        )
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


def is_populated(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    return True


def construct_single_turn_content(test_case: LLMTestCase) -> str:
    text = ""
    for param, display in SINGLE_TURN_CONTENT_PARAMS.items():
        value = getattr(test_case, param.value, None)
        if not is_populated(value):
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
    for param, display in MULTI_TURN_CONTENT_PARAMS.items():
        value = getattr(test_case, param.value, None)
        if not is_populated(value):
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
        if not is_populated(value):
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
            if isinstance(value, list) and not classifier.multi_label:
                raise ValueError(
                    f"expected_labels['{key}'] is a list, but classifier '{key}' assigns one label; set multi_label=True or expect a single label."
                )
            if classifier.label_discovery_config is not None:
                continue
            for label in value if isinstance(value, list) else [value]:
                if label not in classifier.labels:
                    raise ValueError(
                        f"expected_labels['{key}'] = '{label}' is not one of classifier '{key}' labels: {', '.join(str(declared) for declared in classifier.labels)}."
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
