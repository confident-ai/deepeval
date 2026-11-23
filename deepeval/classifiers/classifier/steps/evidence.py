import json
import re
from typing import TYPE_CHECKING, Any, List, Optional, Union

from deepeval.classifiers.classifier.config import EvidenceConfig
from deepeval.classifiers.classifier.schema import EvidenceItem
from deepeval.classifiers.classifier.steps.truncate import ShownContent
from deepeval.classifiers.utils import (
    MULTI_TURN_CONTENT_PARAMS,
    SINGLE_TURN_CONTENT_PARAMS,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall
from deepeval.test_run.api import Evidence

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import ClassificationPrompt

_MIN_EVIDENCE_QUOTE_CHARS = 8
_WHITESPACE_RE = re.compile(r"\s+")
_TRUNCATION_TAIL_RE = re.compile(r"""…?\[truncated\][\s"'\\]*$""")
_TRUNCATION_CLAIM_RE = re.compile(
    r"truncat|cut[ -]?off|cut short|incomplete (?:output|response|content)",
    re.IGNORECASE,
)
TRUNCATION_MARKER_REASON = (
    'The only problem found was the "…[truncated]" marker added where long '
    "fields are shortened, which is not part of the application's output."
)


_TURN_FIELDS = ["content", "retrieval_context", "tools_called"]
_TRACE_FIELDS = [
    "input",
    "output",
    "error",
    "tools_called",
    "retrieval_context",
    "context",
]


def citable_fields(
    test_case: Union[LLMTestCase, ConversationalTestCase],
) -> List[str]:
    if isinstance(test_case, ConversationalTestCase):
        return _TURN_FIELDS + [
            param.value
            for param in MULTI_TURN_CONTENT_PARAMS
            if getattr(test_case, param.value, None)
        ]
    if test_case._trace_dict is not None:
        return _TRACE_FIELDS
    return [
        param.value
        for param in SINGLE_TURN_CONTENT_PARAMS
        if getattr(test_case, param.value, None)
    ]


def blames_truncation_marker(
    label: str,
    description: Optional[str],
    reason: Optional[str],
    evidence_items: List[EvidenceItem],
) -> bool:
    """Whether a label about cut-off content rests on the marker added where
    long fields are shortened, rather than on the application's own output."""
    claims_truncation = _TRUNCATION_CLAIM_RE.search(
        f"{label} {description or ''} {reason or ''}"
    )
    cites_marker = "[truncated]" in (reason or "") or any(
        _TRUNCATION_TAIL_RE.search(evidence_item.quote.strip())
        for evidence_item in evidence_items
    )
    return bool(claims_truncation) and cites_marker


def verify_evidence(
    evidence_items: List[EvidenceItem],
    test_case: Union[LLMTestCase, ConversationalTestCase],
    prompt: "ClassificationPrompt",
    evidence_config: EvidenceConfig,
) -> List[Evidence]:
    dropped_turn_count = (
        len(test_case.turns) - len(prompt.shown)
        if isinstance(test_case, ConversationalTestCase)
        and isinstance(prompt.shown, list)
        else 0
    )
    verified_evidence = []
    for evidence_item in evidence_items:
        if len(verified_evidence) == evidence_config.max_items:
            break
        quote = _TRUNCATION_TAIL_RE.sub(
            "", evidence_item.quote.strip()
        ).rstrip()[: evidence_config.max_quote_chars]
        if not quote:
            continue
        source_texts = (
            _get_source_texts(evidence_item, test_case, prompt.shown)
            if evidence_item.field in prompt.evidence_fields
            else None
        )
        matched_quote = (
            _find_quote_in_sources(quote, source_texts)
            if source_texts and len(quote) >= _MIN_EVIDENCE_QUOTE_CHARS
            else None
        )
        verified_evidence.append(
            Evidence(
                field=evidence_item.field,
                quote=matched_quote or quote,
                note=evidence_item.note.strip(),
                span_index=evidence_item.span_index,
                turn_index=(
                    None
                    if evidence_item.turn_index is None
                    else evidence_item.turn_index + dropped_turn_count
                ),
                matched=matched_quote is not None,
            )
        )
    return verified_evidence


def _find_quote_in_sources(
    quote: str, source_texts: List[str]
) -> Optional[str]:
    quote_variants = [quote]
    unescaped = (
        quote.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
    )
    if unescaped != quote:
        quote_variants.append(unescaped)
    for quote_variant in quote_variants:
        if any(quote_variant in source for source in source_texts):
            return quote_variant
        quote_without_whitespace = _WHITESPACE_RE.sub("", quote_variant)
        if quote_without_whitespace and any(
            quote_without_whitespace in _WHITESPACE_RE.sub("", source)
            for source in source_texts
        ):
            return quote_variant
    return None


def _get_source_texts(
    evidence_item: EvidenceItem,
    test_case: Union[LLMTestCase, ConversationalTestCase],
    shown: Optional[ShownContent],
) -> Optional[List[str]]:
    if isinstance(test_case, ConversationalTestCase):
        if evidence_item.turn_index is None:
            return _to_texts(getattr(test_case, evidence_item.field, None))
        if not isinstance(shown, list) or not (
            1 <= evidence_item.turn_index <= len(shown)
        ):
            return None
        return _to_texts(
            shown[evidence_item.turn_index - 1].get(evidence_item.field)
        )
    if test_case._trace_dict is None:
        return _to_texts(getattr(test_case, evidence_item.field, None))
    if isinstance(shown, str):
        return [shown] if evidence_item.span_index is None else None
    if not isinstance(shown, dict):
        return None
    if evidence_item.span_index is None:
        return _to_texts(shown.get(evidence_item.field))
    span = next(
        (
            s
            for s in shown.get("spans", [])
            if s["index"] == evidence_item.span_index
        ),
        None,
    )
    return None if span is None else _to_texts(span.get(evidence_item.field))


def _to_texts(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts = [
            repr(v) if isinstance(v, ToolCall) else v
            for v in value
            if isinstance(v, (str, ToolCall))
        ]
        return [*texts, str(value), _to_json(value)]
    return [str(value), _to_json(value)]


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
