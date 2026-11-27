from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

_BLEU_TYPES = ("bleu1", "bleu2", "bleu3", "bleu4")
_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


class BleuMetric(BaseMetric):
    """Classic BLEU overlap between answer and reference.

    Wraps ``Scorer.sentence_bleu_score`` as a metric class. Scores the
    n-gram overlap of ``actual_output`` against ``expected_output``.
    Fully **deterministic** (requires the ``nltk`` package and its
    ``punkt`` data). Maintainers favor judged metrics for open-ended
    text; use BLEU for translation-style tasks with tight references.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        bleu_type: str = "bleu1",
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if bleu_type not in _BLEU_TYPES:
            raise ValueError(
                f"Unknown bleu_type '{bleu_type}'. "
                f"Supported: {list(_BLEU_TYPES)}."
            )
        self.bleu_type = bleu_type
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
            try:
                from deepeval.scorer.scorer import Scorer
            except ImportError as exc:
                raise ValueError(f"Cannot load Scorer: {exc}")
            try:
                import nltk  # noqa: F401
            except ImportError:
                raise ValueError(
                    "BleuMetric requires the `nltk` package. Install "
                    "it with `pip install nltk` and download the "
                    "tokenizer with `nltk.download('punkt')`."
                )
            try:
                score = float(
                    Scorer.sentence_bleu_score(
                        test_case.expected_output,
                        test_case.actual_output,
                        self.bleu_type,
                    )
                )
            except LookupError as exc:
                raise ValueError(
                    "BleuMetric needs nltk tokenizer data "
                    f"({exc}). Run `nltk.download('punkt')`."
                )
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(score)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Type: {self.bleu_type}",
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

    def _generate_reason(self, score: float) -> Optional[str]:
        if not self.include_reason:
            return None
        return f"{self.bleu_type} overlap is {score:.2f}."

    @property
    def __name__(self):
        return "BLEU"


class RougeMetric(BaseMetric):
    """Classic ROUGE overlap between answer and reference.

    Wraps ``Scorer.rouge_score`` as a metric class. Fully
    **deterministic** (requires the ``rouge-score`` package).
    Maintainers favor judged metrics for open-ended text; use ROUGE
    for summarization-style tasks with tight references.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        score_type: str = "rougeL",
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if score_type not in _ROUGE_TYPES:
            raise ValueError(
                f"Unknown score_type '{score_type}'. "
                f"Supported: {list(_ROUGE_TYPES)}."
            )
        self.score_type = score_type
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
            try:
                import rouge_score  # noqa: F401
            except ImportError:
                raise ValueError(
                    "RougeMetric requires the `rouge-score` package. "
                    "Install it with `pip install rouge-score`."
                )
            try:
                from deepeval.scorer.scorer import Scorer
            except ImportError as exc:
                raise ValueError(f"Cannot load Scorer: {exc}")
            score = float(
                Scorer.rouge_score(
                    test_case.expected_output,
                    test_case.actual_output,
                    self.score_type,
                )
            )
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(score)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Type: {self.score_type}",
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

    def _generate_reason(self, score: float) -> Optional[str]:
        if not self.include_reason:
            return None
        return f"{self.score_type} overlap is {score:.2f}."

    @property
    def __name__(self):
        return "ROUGE"


class BertScoreMetric(BaseMetric):
    """Classic BERTScore overlap between answer and reference.

    Wraps ``Scorer.bert_score`` (mean F1) as a metric class. Fully
    **deterministic** given the same model weights, but requires the
    ``bert_score`` and ``torch`` packages plus a model download, so it
    is far heavier than the other overlap metrics.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.model_name = model_name
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
            try:
                import bert_score  # noqa: F401
                import torch  # noqa: F401
            except ImportError:
                raise ValueError(
                    "BertScoreMetric requires `bert_score` and `torch`. "
                    "Install them with `pip install bert-score torch`."
                )
            try:
                from deepeval.scorer.scorer import Scorer
            except ImportError as exc:
                raise ValueError(f"Cannot load Scorer: {exc}")
            result = Scorer.bert_score(
                test_case.expected_output,
                test_case.actual_output,
                model=self.model_name,
            )
            f1_values = result.get("bert-f1", [0.0])
            score = float(sum(f1_values) / len(f1_values))
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(score)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Model: {self.model_name}",
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

    def _generate_reason(self, score: float) -> Optional[str]:
        if not self.include_reason:
            return None
        return f"BERTScore F1 is {score:.2f}."

    @property
    def __name__(self):
        return "BERTScore"


class PassAtKMetric(BaseMetric):
    """Probability that at least one of k samples passes (pass@k).

    Wraps ``Scorer.pass_at_k`` as a metric class: given ``n`` sampled
    solutions of which ``c`` are correct, the chance that a random
    subset of size ``k`` contains a correct one. Pass ``n``/``c``
    directly or via ``test_case.metadata`` (``pass_at_k_n`` /
    ``pass_at_k_c``). Fully **deterministic**.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        k: int = 1,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if k < 1:
            raise ValueError(f"`k` must be at least 1, got {k}.")
        self.k = k
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
        n: Optional[int] = None,
        c: Optional[int] = None,
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
            metadata = test_case.metadata or {}
            if n is None:
                n = metadata.get("pass_at_k_n")
            if c is None:
                c = metadata.get("pass_at_k_c")
            self._validate(n, c)
            try:
                import numpy  # noqa: F401
            except ImportError:
                raise ValueError(
                    "PassAtKMetric requires `numpy`. Install it with "
                    "`pip install numpy`."
                )
            try:
                from deepeval.scorer.scorer import Scorer
            except ImportError as exc:
                raise ValueError(f"Cannot load Scorer: {exc}")
            score = float(Scorer().pass_at_k(n, c, self.k))
            self.score_breakdown = {
                "pass_at_k": score,
                "n": float(n),
                "c": float(c),
                "k": float(self.k),
            }
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(n, c, score)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"n={n}, c={c}, k={self.k}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        n: Optional[int] = None,
        c: Optional[int] = None,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            n=n,
            c=c,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _validate(self, n, c):
        if not isinstance(n, int) or not isinstance(c, int):
            raise ValueError(
                "PassAtKMetric needs integer `n` (samples) and `c` "
                "(correct). Pass them to `measure` or set "
                "`metadata={'pass_at_k_n': n, 'pass_at_k_c': c}`."
            )
        if n < 1 or c < 0 or c > n:
            raise ValueError(f"Need 0 <= c <= n with n >= 1, got n={n}, c={c}.")
        if self.k > n:
            raise ValueError(f"`k` ({self.k}) cannot exceed `n` ({n}).")

    def _generate_reason(self, n: int, c: int, score: float) -> Optional[str]:
        if not self.include_reason:
            return None
        return (
            f"pass@{self.k} is {score:.2f} with {c} correct "
            f"of {n} sampled solution(s)."
        )

    @property
    def __name__(self):
        return "Pass At K"
