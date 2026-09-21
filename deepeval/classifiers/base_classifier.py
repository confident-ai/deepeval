from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel

from deepeval.metrics.base_metric import PromptMixin
from deepeval.test_case import ConversationalTestCase, LLMTestCase

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM


# Sentinel label the judge returns when none of the declared labels fit.
NONE_LABEL = "NONE"


class Label(BaseModel):
    """A label a classifier can assign. The description is the boundary for
    the label, the same way G-Eval criteria are the boundary for a score."""

    name: str
    description: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        super().__init__(**kwargs)


class BaseClassifier(PromptMixin):
    """Classifier counterpart of ``BaseMetric``.

    A classifier assigns exactly one label from a closed set to a test case.
    It only produces a pass/fail verdict when the test case declares an
    expected label for it (``test_case.expected_labels[classifier.name]``);
    otherwise ``success`` is ``None``, mirroring a metric with no threshold.
    """

    _template_feature = "classifiers"

    name: str
    labels: List[Label]
    label: Optional[str] = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    # Kept only so shared helpers written for metrics (e.g. the progress
    # indicator description) can read it. Classifiers have no strict mode.
    strict_mode: bool = False
    async_mode: bool = True
    include_reason: bool = True
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    skipped = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def classify(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        *args,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def a_classify(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        *args,
        **kwargs,
    ) -> str:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    def is_successful(
        self, expected_label: Optional[str] = None
    ) -> Optional[bool]:
        if expected_label is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            self.success = self.label == expected_label
        return self.success

    @property
    def label_names(self) -> List[str]:
        return [label.name for label in self.labels]

    @property
    def __name__(self):
        return self.name

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
