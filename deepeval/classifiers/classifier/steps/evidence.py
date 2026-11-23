import json
import re
from typing import TYPE_CHECKING, Any, List, Optional, Union

from deepeval.classifiers.classifier.config import EvidenceConfig
from deepeval.classifiers.classifier.schema import EvidenceItem
from deepeval.classifiers.classifier.steps.truncate import ShownContent
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall
from deepeval.test_run.api import Evidence

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import ClassificationPrompt

_MIN_EVIDENCE_QUOTE_CHARS = 8
_WHITESPACE_RE = re.compile(r"\s+")
_TRUNCATION_TAIL_RE = re.compile(r"""…?\[truncated\][\s"'\\]*$""")


def verify_evidence(
    items: List[EvidenceItem],
    test_case: Union[LLMTestCase, ConversationalTestCase],
    prompt: "ClassificationPrompt",
    config: EvidenceConfig,
) -> List[Evidence]:
    turn_offset = (
        len(test_case.turns) - len(prompt.shown)
        if isinstance(test_case, ConversationalTestCase)
        and isinstance(prompt.shown, list)
        else 0
    )
    verified = []
    for item in items:
        if len(verified) == config.max_items:
            break
        quote = _TRUNCATION_TAIL_RE.sub("", item.quote.strip()).rstrip()[
            : config.max_quote_chars
        ]
        if not quote:
            continue
        sources = (
            _sources(item, test_case, prompt.shown)
            if item.field in prompt.evidence_fields
            else None
        )
        matched = (
            _match_quote(quote, sources)
            if sources and len(quote) >= _MIN_EVIDENCE_QUOTE_CHARS
            else None
        )
        verified.append(
            Evidence(
                field=item.field,
                quote=matched or quote,
                note=item.note.strip(),
                span_index=item.span_index,
                turn_index=(
                    None
                    if item.turn_index is None
                    else item.turn_index + turn_offset
                ),
                matched=matched is not None,
            )
        )
    return verified


def _match_quote(quote: str, sources: List[str]) -> Optional[str]:
    candidates = [quote]
    unescaped = (
        quote.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
    )
    if unescaped != quote:
        candidates.append(unescaped)
    for candidate in candidates:
        if any(candidate in source for source in sources):
            return candidate
        compact = _WHITESPACE_RE.sub("", candidate)
        if compact and any(
            compact in _WHITESPACE_RE.sub("", source) for source in sources
        ):
            return candidate
    return None


def _sources(
    item: EvidenceItem,
    test_case: Union[LLMTestCase, ConversationalTestCase],
    shown: Optional[ShownContent],
) -> Optional[List[str]]:
    if isinstance(test_case, ConversationalTestCase):
        if item.turn_index is None:
            return _texts(getattr(test_case, item.field, None))
        if not isinstance(shown, list) or not (
            1 <= item.turn_index <= len(shown)
        ):
            return None
        return _texts(shown[item.turn_index - 1].get(item.field))
    if test_case._trace_dict is None:
        return _texts(getattr(test_case, item.field, None))
    if isinstance(shown, str):
        return [shown] if item.span_index is None else None
    if not isinstance(shown, dict):
        return None
    if item.span_index is None:
        return _texts(shown.get(item.field))
    span = next(
        (s for s in shown.get("spans", []) if s["index"] == item.span_index),
        None,
    )
    return None if span is None else _texts(span.get(item.field))


def _texts(value: Any) -> Optional[List[str]]:
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
        return [*texts, str(value), _dumps(value)]
    return [str(value), _dumps(value)]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
