"""LLM-judged classifier: assigns exactly one label from a closed set."""

import asyncio
from typing import List, Optional, Sequence, Tuple, Type, Union

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.classifiers.classifier.schema import ClassificationResult
from deepeval.classifiers.utils import (
    construct_multi_turn_content,
    construct_single_turn_content,
    construct_turns,
    format_labels,
    normalize_labels,
    resolve_label,
)
from deepeval.config.settings import get_settings
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.utils import get_or_create_event_loop

ClassifierTemplate = make_template_class("Classifier", feature="classifiers")


class Classifier(BaseClassifier):
    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        if not name or not name.strip():
            raise ValueError("Classifier 'name' cannot be empty.")

        self.name = name
        self.labels = normalize_labels(labels)
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.evaluation_template = evaluation_template

    def classify(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> str:
        self._check_test_case(test_case)
        self._reset_usage()

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                settings = get_settings()
                loop.run_until_complete(
                    asyncio.wait_for(
                        self.a_classify(
                            test_case,
                            _show_indicator=False,
                            _in_component=_in_component,
                        ),
                        timeout=(
                            None
                            if settings.DEEPEVAL_DISABLE_TIMEOUTS
                            else settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS
                        ),
                    )
                )
            else:
                prompt = self._build_prompt(test_case)
                raw_label, reason = generate_with_schema_and_extract(
                    self,
                    prompt,
                    ClassificationResult,
                    extract_schema=lambda r: (r.label, r.reason),
                    extract_json=lambda d: (d.get("label"), d.get("reason")),
                )
                self._finalize(raw_label, reason)

            return self.label

    async def a_classify(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> str:
        self._check_test_case(test_case)
        self._reset_usage()

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            prompt = self._build_prompt(test_case)
            raw_label, reason = await a_generate_with_schema_and_extract(
                self,
                prompt,
                ClassificationResult,
                extract_schema=lambda r: (r.label, r.reason),
                extract_json=lambda d: (d.get("label"), d.get("reason")),
            )
            self._finalize(raw_label, reason)
            return self.label

    ###############################################
    # Internals
    ###############################################

    def _check_test_case(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> None:
        if not isinstance(test_case, (LLMTestCase, ConversationalTestCase)):
            raise TypeError(
                f"{self.__class__.__name__} expects an LLMTestCase or ConversationalTestCase, got {type(test_case).__name__}."
            )

    def _reset_usage(self) -> None:
        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        self.label = None
        self.reason = None
        self.error = None

    def _build_prompt(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> str:
        labels = format_labels(self.labels)
        if isinstance(test_case, ConversationalTestCase):
            return self._get_prompt(
                "classify_multi_turn",
                labels=labels,
                test_case_content=construct_multi_turn_content(test_case),
                turns=construct_turns(test_case),
            )
        return self._get_prompt(
            "classify_single_turn",
            labels=labels,
            test_case_content=construct_single_turn_content(test_case),
        )

    def _finalize(
        self, raw_label: Optional[str], reason: Optional[str]
    ) -> None:
        label = resolve_label(raw_label, self.labels)
        if label is None:
            self.error = (
                f"Judge returned '{raw_label}', which is not one of the declared labels "
                f"({', '.join(self.label_names)}) or {NONE_LABEL}."
            )
            self.label = None
        else:
            self.label = label
        self.reason = reason if self.include_reason else None
