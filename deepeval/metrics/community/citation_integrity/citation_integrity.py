import re
from typing import List, Optional, Set

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
)

# `[...]` not followed by `(`, so markdown links are not read as citations.
_MARKER_RE = re.compile(r"\[([^\[\]]+)\](?!\()")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_MARKERS_RE = re.compile(r"^((?:\s*\[[^\[\]]+\](?!\())+)\s*(.*)$")


class CitationIntegrityMetric(BaseMetric):
    """Is every sentence cited, and does every citation point to a real passage?

    Checks the *structure* of citations in ``actual_output`` against
    ``retrieval_context``, without an LLM:

    - A citation is a ``[...]`` marker holding one or more comma-separated
      references. ``[N]`` refers to the N-th passage (1-based); any other
      reference must equal the ``source`` of a ``RetrievedContextData``
      passage. A reference that resolves to neither is **broken** (e.g.
      ``[7]`` when only 3 passages were retrieved, or an invented source).
    - A sentence is **cited** when it carries at least one citation and all
      of its references resolve.

    The score is the fraction of cited sentences, or ``0`` if any citation
    is broken, so invented sources always fail. This metric does not judge
    whether a cited passage *supports* its claim; pair it with
    ``CitationFaithfulnessMetric`` for that.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            None,
            test_case.multimodal,
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            passage_count = len(test_case.retrieval_context)
            sources = {
                passage.source
                for passage in test_case.retrieval_context
                if isinstance(passage, RetrievedContextData)
            }
            sentences = self._split_sentences(test_case.actual_output)

            broken: List[str] = []
            uncited: List[str] = []
            for sentence in sentences:
                references = self._references(sentence)
                invalid = [
                    reference
                    for reference in references
                    if not self._resolves(reference, passage_count, sources)
                ]
                broken.extend(invalid)
                if not references or invalid:
                    uncited.append(sentence)

            cited_count = len(sentences) - len(uncited)
            score = 0.0 if broken else cited_count / len(sentences)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(
                broken, cited_count, len(sentences), passage_count
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Passages in retrieval context: {passage_count}",
                    f"Sources: {sorted(sources) if sources else []}",
                    f"Broken references: {broken}",
                    f"Uncited sentences:\n{uncited}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences: List[str] = []
        for fragment in _SENTENCE_SPLIT_RE.split(text.strip()):
            fragment = fragment.strip()
            # "Claim. [1] Next claim." splits "[1]" onto the next fragment;
            # leading markers belong to the previous sentence.
            leading = _LEADING_MARKERS_RE.match(fragment)
            if sentences and leading:
                sentences[-1] = f"{sentences[-1]} {leading.group(1).strip()}"
                fragment = leading.group(2).strip()
            if fragment and any(char.isalnum() for char in fragment):
                sentences.append(fragment)
        return sentences

    @staticmethod
    def _references(sentence: str) -> List[str]:
        return [
            reference.strip()
            for marker in _MARKER_RE.findall(sentence)
            for reference in marker.split(",")
            if reference.strip()
        ]

    @staticmethod
    def _resolves(
        reference: str, passage_count: int, sources: Set[str]
    ) -> bool:
        if reference.isdigit():
            return 1 <= int(reference) <= passage_count
        return reference in sources

    def _generate_reason(
        self,
        broken: List[str],
        cited_count: int,
        sentence_count: int,
        passage_count: int,
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        if broken:
            markers = ", ".join(f"[{reference}]" for reference in broken)
            return (
                f"Found broken citation(s) {markers} that match no passage "
                f"in the retrieval context ({passage_count} passage(s))."
            )
        if cited_count == sentence_count:
            return (
                f"All {sentence_count} sentence(s) carry a citation to a "
                "passage in the retrieval context."
            )
        return (
            f"{cited_count} of {sentence_count} sentence(s) carry a citation; "
            f"{sentence_count - cited_count} sentence(s) are uncited."
        )

    @property
    def __name__(self):
        return "Citation Integrity"
