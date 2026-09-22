from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional, Dict, List, Literal, Tuple

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    SingleTurnParams,
    ArenaTestCase,
)
from deepeval.templates.resolver import (
    MetricTemplateMethod,
    resolve_template,
)
from deepeval.templates.template_class import filter_template_kwargs

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel


###############################################
# QAG (question-answer generation) verdicts
###############################################
#
# Every yes/no-style metric asks an LLM judge to classify a list of items
# (claims, statements, opinions, ...) into one of two vocabularies:
#
# - ``YesNo``: a strict binary verdict.
# - ``YesNoBorderline``: a binary verdict plus a ``borderline`` bucket for
#   items that are ambiguous / only partially supported. Each metric decides
#   how the borderline bucket is scored (see ``score_qag_verdicts`` in
#   ``deepeval.metrics.utils``).
#
# Always refer to verdicts through the ``Verdict`` enum (``Verdict.YES``)
# rather than raw strings so a typo is an ``AttributeError`` at import time
# instead of a silently wrong score. Members subclass ``str`` so
# ``Verdict.YES == "yes"`` and they serialize as plain strings.


class Verdict(str, Enum):
    YES = "yes"
    NO = "no"
    BORDERLINE = "borderline"

    def __str__(self) -> str:
        return self.value


# Field types for pydantic schemas. Pydantic renders these as
# ``{"enum": ["yes", "no"], "type": "string"}`` and coerces incoming strings to
# ``Verdict`` members.
YesNo = Literal[Verdict.YES, Verdict.NO]
YesNoBorderline = Literal[Verdict.YES, Verdict.NO, Verdict.BORDERLINE]

# Allowed-vocabulary tuples for ``generate_qag_verdicts(..., allowed=...)``.
YES_NO: Tuple[Verdict, ...] = (Verdict.YES, Verdict.NO)
YES_NO_BORDERLINE: Tuple[Verdict, ...] = (
    Verdict.YES,
    Verdict.NO,
    Verdict.BORDERLINE,
)

# Older prompts used ``idk`` for the borderline bucket. Judges that saw a cached
# or third-party copy of those prompts may still reply with it.
LEGACY_VERDICT_ALIASES: Dict[str, Verdict] = {"idk": Verdict.BORDERLINE}


class PromptMixin:
    """Renders a metric prompt template. `template_class` overrides the default
    `self.__class__.__name__` when borrowing another class's templates.
    `_template_feature` selects the `templates/<feature>/templates.json`
    bundle and `_template_attr` names the instance attribute holding a
    user-supplied template class; metrics use the defaults, classifiers
    override both."""

    _template_feature: str = "metrics"
    _template_attr: str = "evaluation_template"

    def _get_prompt(
        self,
        method: MetricTemplateMethod,
        *,
        template_class: Optional[str] = None,
        multimodal: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> str:
        context = {**kwargs, "multimodal": multimodal, "strict": strict}

        # An explicit `template_class` borrows another class's templates, so a
        # user template set for this metric must not hijack it.
        if template_class is None:
            render = getattr(
                getattr(self, self._template_attr, None), method, None
            )
            if render is not None:
                return render(**filter_template_kwargs(render, context))

        return resolve_template(
            self._template_feature,
            template_class or self.__class__.__name__,
            method,
            **context,
        )


class BaseMetric(PromptMixin):
    _required_params = List[SingleTurnParams]
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    requires_trace: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None
    system_one_model: Optional[DeepEvalBaseSystemOneModel] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


class BaseConversationalMetric(PromptMixin):
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None
    system_one_model: Optional[DeepEvalBaseSystemOneModel] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Conversational Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


class BaseArenaMetric(PromptMixin):
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(self, test_case: ArenaTestCase, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(self, test_case: ArenaTestCase, *args, **kwargs) -> str:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Arena Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens
