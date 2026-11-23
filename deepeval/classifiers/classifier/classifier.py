"""LLM-judged classifier: assigns exactly one label from a closed set."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.classifiers.classifier.schema import (
    LABEL_LIST_SCHEMAS,
    ClassificationResult,
    EvidencedClassificationResult,
    LabelReview,
    SuggestedClassificationResult,
    SuggestedEvidencedClassificationResult,
)
from deepeval.classifiers.classifier.config import (
    ClassificationContext,
    EvidenceConfig,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.classifiers.classifier.steps.evidence import verify_evidence
from deepeval.test_run.api import Classification, Evidence
from deepeval.classifiers.classifier.steps.label_discovery import (
    LabelPool,
    a_merge_new_labels,
    is_new_label,
    merge_new_labels,
    merged_label,
)
from deepeval.classifiers.classifier.steps.review import (
    a_review_classifications,
    review_classifications,
)
from deepeval.classifiers.classifier.steps.truncate import (
    MAX_CONTENT_CHARS,
    fit_text,
    fit_trace,
    fit_turns,
)
from deepeval.classifiers.utils import (
    SystemOneClassifySpec,
    a_generate_classification,
    construct_multi_turn_content,
    construct_multi_turn_state,
    construct_single_turn_content,
    construct_single_turn_state,
    construct_trace_content,
    construct_trace_dict,
    construct_turns,
    MULTI_TURN_CONTENT_PARAMS,
    SINGLE_TURN_CONTENT_PARAMS,
    clean_reason,
    format_labels,
    generate_classification,
    normalize_labels,
    resolve_label,
    resolve_suggested_label,
)
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
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

_TRUNCATION_MARKER_REASON = (
    'The only problem found was the "…[truncated]" marker added where long '
    "fields are shortened, which is not part of the application's output."
)

_RESULT_SCHEMAS = {
    (True, False): SuggestedClassificationResult,
    (False, True): EvidencedClassificationResult,
    (True, True): SuggestedEvidencedClassificationResult,
}
_EVIDENCE_TURN_FIELDS = ["content", "retrieval_context", "tools_called"]
_EVIDENCE_TRACE_FIELDS = [
    "input",
    "output",
    "error",
    "toolsCalled",
    "retrievalContext",
    "context",
]

ClassifierTemplate = make_template_class("Classifier", feature="classifiers")

# Built-in classifiers (``RefusalClassifier`` ...) share ``Classifier``'s bundle
# entry. Public templates reach it through ``classification_template``; the
# ``_experimental_*`` ones are not exposed on the template class, so they name
# the bundle key explicitly.
_EXPERIMENTAL_TEMPLATE_CLASS = "Classifier"

logger = logging.getLogger(__name__)


class Classifier(BaseClassifier):

    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]],
        description: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[ClassifierEvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
        allow_none: bool = False,
        multi_label: bool = False,
        max_content_chars: int = MAX_CONTENT_CHARS,
        evidence_config: Optional[EvidenceConfig] = None,
        review_config: Optional[ReviewConfig] = None,
        label_discovery_config: Optional[LabelDiscoveryConfig] = None,
        classification_context: Optional[ClassificationContext] = None,
    ):
        if not name or not name.strip():
            raise ValueError("Classifier 'name' cannot be empty.")

        self.name = name
        self.description = description
        self.classification_context = classification_context
        self.labels = (
            normalize_labels(labels)
            if labels or label_discovery_config is None
            else []
        )
        self.eval_mode = resolve_classifier_eval_mode(eval_mode)
        if (
            label_discovery_config is not None
            or evidence_config is not None
            or review_config is not None
            or multi_label
        ) and self.eval_mode.uses_system_one:
            raise ValueError(
                "label_discovery_config, evidence_config, review_config and multi_label need an LLM judge; they are not supported in the system_one eval mode."
            )
        self.review_config = review_config
        self.multi_label = multi_label
        self.max_content_chars = max_content_chars
        self._shown = None
        self._review_content = ("test case", "")
        self.label_discovery_config = label_discovery_config
        self.evidence_config = evidence_config
        self.label_pool = LabelPool(
            normalize_labels(label_discovery_config.candidate_labels)
            if label_discovery_config
            and label_discovery_config.candidate_labels
            else []
        )
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
                schema = self._result_schema()
                if schema is not None:
                    result = generate_with_schema_and_extract(
                        metric=self,
                        prompt=prompt,
                        schema_cls=schema,
                        extract_schema=lambda result: result,
                        extract_json=schema.model_validate,
                    )
                    merges = merge_new_labels(self, result)
                    self._build_classifications(result, test_case, merges)
                else:
                    raw_label, reason = generate_classification(
                        self,
                        prompt,
                        schema_cls=ClassificationResult,
                        system_one=self._experimental_system_one_spec(
                            test_case
                        ),
                    )
                    self._build_classification(raw_label, reason)
                if self.review_config:
                    review_classifications(self)

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
            schema = self._result_schema()
            if schema is not None:
                result = await a_generate_with_schema_and_extract(
                    metric=self,
                    prompt=prompt,
                    schema_cls=schema,
                    extract_schema=lambda result: result,
                    extract_json=schema.model_validate,
                )
                merges = await a_merge_new_labels(self, result)
                self._build_classifications(result, test_case, merges)
            else:
                raw_label, reason = await a_generate_classification(
                    self,
                    prompt,
                    schema_cls=ClassificationResult,
                    system_one=self._experimental_system_one_spec(test_case),
                )
                self._build_classification(raw_label, reason)
            if self.review_config:
                await a_review_classifications(self)
            return self.label

    def is_successful(
        self, expected_label: Optional[str] = None
    ) -> Optional[bool]:
        if not self.multi_label or expected_label is None or self.error:
            return super().is_successful(expected_label)
        self.success = expected_label in [r.label for r in self.classifications]
        return self.success

    def copy(self) -> "Classifier":
        clone = super().copy()
        clone.label_pool = self.label_pool
        return clone

    ###############################################
    # Classification
    ###############################################

    def _build_classifications(
        self,
        result: Any,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        merges: Dict[str, Label],
    ) -> None:
        items = result.labels if self.multi_label else [result]
        none_reason = None
        unresolved = []
        for item in items:
            label, description = self._resolve(
                item.label,
                item.reason,
                getattr(item, "new_label_description", None),
            )
            if label == NONE_LABEL:
                none_reason = none_reason or clean_reason(item.reason)
                continue
            if label is None:
                unresolved.append(item.label)
                continue
            if label in merges:
                label, description = merged_label(self, merges[label])
            elif description is not None and is_new_label(self, label):
                continue
            evidence = (
                verify_evidence(
                    getattr(item, "evidence", []),
                    test_case,
                    self._shown,
                    self.evidence_config.max_quote_chars,
                )
                if self.evidence_config
                else []
            )
            if self._rejects_evidence(evidence):
                none_reason = none_reason or _TRUNCATION_MARKER_REASON
                continue
            kept = next(
                (r for r in self.classifications if r.label == label), None
            )
            if kept is not None:
                kept.evidence.extend(evidence)
                continue
            self.classifications.append(
                Classification(
                    name=self.name,
                    label=label,
                    reason=(
                        clean_reason(item.reason)
                        if self.include_reason
                        else None
                    ),
                    new_label_description=description,
                    evidence=evidence,
                )
            )
        top = self.classifications[0] if self.classifications else None
        self.label = top.label if top else None
        reason = top.reason if top else none_reason
        self.reason = reason if self.include_reason else None
        if top is None and (unresolved or not self.allow_none):
            self.error = self._unresolved_error(
                unresolved[0] if unresolved else None
            )

    def _build_classification(
        self, raw_label: Optional[str], reason: Optional[str]
    ) -> None:
        reason = clean_reason(reason) if reason else reason
        label, _ = self._resolve(raw_label, reason, None)
        if label is None:
            self.error = self._unresolved_error(raw_label)
            self.label = None
        elif label == NONE_LABEL:
            # The judge declined to classify; that is a valid outcome when
            # allow_none is on, surfaced to users as `label is None` with no error.
            self.label = None
        else:
            self.label = label
            self.classifications = [
                Classification(
                    name=self.name,
                    label=label,
                    reason=reason if self.include_reason else None,
                )
            ]
        self.reason = reason if self.include_reason else None

    def _resolve(
        self,
        raw_label: Optional[str],
        reason: Optional[str],
        new_label_description: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.label_discovery_config:
            return resolve_suggested_label(
                raw_label,
                new_label_description,
                reason,
                self.labels,
                self.candidate_labels,
                self.allow_none,
            )
        return resolve_label(raw_label, self.labels, self.allow_none), None

    def _unresolved_error(self, raw_label: Optional[str]) -> str:
        allowed = ", ".join(str(label) for label in self.labels)
        if self.allow_none:
            allowed += ", or no classification"
        return (
            f"Judge returned '{raw_label}', which is not one of the declared labels "
            f"({allowed})."
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

    ###############################################
    # Evidence
    ###############################################

    def _cite_evidence(self, subject: str, fields: List[str]) -> str:
        if not self.evidence_config:
            return ""
        return self._get_prompt(
            "cite_evidence",
            subject=subject,
            fields=", ".join(f'"{field}"' for field in fields),
            allow_none=self.allow_none,
            max_items=self.evidence_config.max_items,
            max_quote_chars=self.evidence_config.max_quote_chars,
        )

    def _rejects_evidence(self, evidence: List[Evidence]) -> bool:
        return False

    ###############################################
    # Label Discovery
    ###############################################

    @property
    def candidate_labels(self) -> List[Label]:
        return self.label_pool.labels

    ###############################################
    # Review
    ###############################################

    def _passes_review(self, review: LabelReview) -> bool:
        return review.confidence >= self.review_config.min_confidence

    def _review_reason(self, review: LabelReview) -> str:
        return review.reason

    def _review_schema(self) -> Type[LabelReview]:
        return LabelReview

    ###############################################
    # Prompt
    ###############################################

    def _build_prompt(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> str:
        labels = format_labels(self.labels)
        extras = {
            "multi_label": self.multi_label,
            "context": self._context_prompt(),
            "reason_rules": self._get_prompt("reason_rules"),
            "guidance": self._guidance_prompt(),
            "suggest_labels": (
                self._get_prompt(
                    "suggest_labels",
                    allow_none=self.allow_none,
                    candidate_labels=bool(self.candidate_labels),
                    label_naming_rule=self._get_prompt("label_naming_rule"),
                )
                if self.label_discovery_config
                else ""
            ),
            "candidate_labels": (
                format_labels(self.candidate_labels)
                if self.label_discovery_config and self.candidate_labels
                else ""
            ),
        }
        if isinstance(test_case, ConversationalTestCase):
            extras["cite_evidence"] = self._cite_evidence(
                "conversation",
                _EVIDENCE_TURN_FIELDS
                + [
                    param.value
                    for param in MULTI_TURN_CONTENT_PARAMS
                    if getattr(test_case, param.value, None)
                ],
            )
            content = construct_multi_turn_content(test_case)
            turns = self._remember_shown(
                fit_turns(construct_turns(test_case), self.max_content_chars)
            )
            self._review_content = (
                "conversation",
                f"{content}\n\nTurns:\n{turns}",
            )
            return self._get_prompt(
                "classify_multi_turn",
                labels=labels,
                test_case_content=content,
                turns=turns,
                allow_none=self.allow_none,
                **extras,
            )
        if test_case._trace_dict is not None:
            extras["cite_evidence"] = self._cite_evidence(
                "trace", _EVIDENCE_TRACE_FIELDS
            )
            trace_content = construct_trace_content(
                self._remember_shown(
                    fit_trace(
                        construct_trace_dict(test_case), self.max_content_chars
                    )
                )
            )
            self._review_content = ("trace", trace_content)
            return self._get_prompt(
                "classify_trace",
                labels=labels,
                trace_content=trace_content,
                allow_none=self.allow_none,
                **extras,
            )
        extras["cite_evidence"] = self._cite_evidence(
            "test case",
            [
                param.value
                for param in SINGLE_TURN_CONTENT_PARAMS
                if getattr(test_case, param.value, None)
            ],
        )
        test_case_content = fit_text(
            construct_single_turn_content(test_case), self.max_content_chars
        )
        self._review_content = ("test case", test_case_content)
        return self._get_prompt(
            "classify_single_turn",
            labels=labels,
            test_case_content=test_case_content,
            allow_none=self.allow_none,
            **extras,
        )

    def _context_prompt(self) -> str:
        context = self.classification_context
        return self._get_prompt(
            "classification_context",
            name=self.name,
            description=self.description,
            has_context=context is not None,
            app_description=context.app_description if context else None,
            trace_description=context.trace_description if context else None,
        )

    def _guidance_prompt(self) -> str:
        return ""

    def _result_schema(self):
        item = _RESULT_SCHEMAS.get(
            (
                self.label_discovery_config is not None,
                self.evidence_config is not None,
            )
        )
        if self.multi_label:
            return LABEL_LIST_SCHEMAS[item or ClassificationResult]
        return item

    def _remember_shown(self, shown: Any) -> Any:
        self._shown = shown
        return shown

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
        self.classifications: List[Classification] = []
        self.error = None
