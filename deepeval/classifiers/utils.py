from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Union

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.metrics.utils.turns import convert_turn_to_dict
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
