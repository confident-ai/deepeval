"""LLM-judged classifier: assigns exactly one label from a closed set."""

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.classifiers.classifier.schema import ClassificationResult
from deepeval.classifiers.utils import (
    SystemOneClassifySpec,
    a_generate_classification,
    construct_multi_turn_content,
    construct_multi_turn_state,
    construct_single_turn_content,
    construct_single_turn_state,
    construct_turns,
    format_labels,
    generate_classification,
    normalize_labels,
    resolve_label,
)
from deepeval.config.settings import get_settings
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    initialize_model,
    initialize_system_one_model,
)
from deepeval.config.eval_mode import (
    ClassifierEvalModeName,
    resolve_classifier_eval_mode,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.utils import get_or_create_event_loop

ClassifierTemplate = make_template_class("Classifier", feature="classifiers")

# Built-in classifiers (``RefusalClassifier`` ...) share ``Classifier``'s bundle
# entry. Public templates reach it through ``classification_template``; the
# ``_experimental_*`` ones are not exposed on the template class, so they name
# the bundle key explicitly.
_EXPERIMENTAL_TEMPLATE_CLASS = "Classifier"


class Classifier(BaseClassifier):
    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[ClassifierEvalModeName] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        if not name or not name.strip():
            raise ValueError("Classifier 'name' cannot be empty.")

        self.name = name
        self.labels = normalize_labels(labels)
        self.eval_mode = resolve_classifier_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.allow_none = allow_none
        self.async_mode = async_mode
        self.classification_template = classification_template

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
                raw_label, reason = generate_classification(
                    self,
                    prompt,
                    schema_cls=ClassificationResult,
                    system_one=self._experimental_system_one_spec(test_case),
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
            raw_label, reason = await a_generate_classification(
                self,
                prompt,
                schema_cls=ClassificationResult,
                system_one=self._experimental_system_one_spec(test_case),
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
                allow_none=self.allow_none,
            )
        return self._get_prompt(
            "classify_single_turn",
            labels=labels,
            test_case_content=construct_single_turn_content(test_case),
            allow_none=self.allow_none,
        )

    def _experimental_system_one_spec(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> SystemOneClassifySpec:
        """Used when the eval mode is `system_one`; see EXPERIMENTAL.md.

        The label is one Choice over the declared labels (plus ``NONE`` when
        ``allow_none``), and the reason is built from the answer.
        """
        multi_turn = isinstance(test_case, ConversationalTestCase)
        state = (
            construct_multi_turn_state(test_case)
            if multi_turn
            else construct_single_turn_state(test_case)
        )
        options: Dict[str, Optional[str]] = {
            label.name: label.description for label in self.labels
        }
        if self.allow_none:
            options[NONE_LABEL] = "None of the other labels apply."

        return SystemOneClassifySpec(
            instructions=self._get_prompt(
                "_experimental_system_one_classify",
                template_class=_EXPERIMENTAL_TEMPLATE_CLASS,
                allow_none=self.allow_none,
                multi_turn=multi_turn,
            ),
            options=options,
            state=state,
            include_reason=self.include_reason,
        )

    def _finalize(
        self, raw_label: Optional[str], reason: Optional[str]
    ) -> None:
        label = resolve_label(raw_label, self.labels, self.allow_none)
        if label is None:
            allowed = ", ".join(str(label) for label in self.labels)
            if self.allow_none:
                allowed += ", or no classification"
            self.error = (
                f"Judge returned '{raw_label}', which is not one of the declared labels "
                f"({allowed})."
            )
            self.label = None
        elif label == NONE_LABEL:
            # The judge declined to classify; that is a valid outcome when
            # allow_none is on, surfaced to users as `label is None` with no error.
            self.label = None
        else:
            self.label = label
        self.reason = reason if self.include_reason else None
