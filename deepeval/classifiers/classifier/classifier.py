"""LLM-judged classifier: assigns one or more labels to a test case."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.classifiers.classifier.config import (
    ClassificationContext,
    EvidenceConfig,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.classifiers.classifier.schema import (
    LABEL_LIST_SCHEMAS,
    ClassificationResult,
    EvidencedClassificationResult,
    LabelReview,
    SuggestedClassificationResult,
    SuggestedEvidencedClassificationResult,
)
from deepeval.classifiers.classifier.steps.evidence import verify_evidence
from deepeval.classifiers.classifier.steps.label_discovery import (
    LabelPool,
    a_merge_new_labels,
    merge_new_labels,
    same_name,
)
from deepeval.classifiers.classifier.steps.review import (
    a_review_classifications,
    review_classifications,
)
from deepeval.classifiers.classifier.steps.trace import (
    construct_trace_content,
    construct_trace_dict,
)
from deepeval.classifiers.classifier.steps.truncate import (
    MAX_CONTENT_CHARS,
    fit_text,
    fit_trace,
    fit_turns,
)
from deepeval.classifiers.utils import (
    MULTI_TURN_CONTENT_PARAMS,
    SINGLE_TURN_CONTENT_PARAMS,
    LabelOrigin,
    ResolvedLabel,
    SystemOneClassifySpec,
    a_generate_classification,
    clean_reason,
    construct_multi_turn_content,
    construct_multi_turn_state,
    construct_single_turn_content,
    construct_single_turn_state,
    construct_turns,
    format_labels,
    generate_classification,
    normalize_labels,
    resolve_label,
    resolve_suggested_label,
)
from deepeval.config.eval_mode import (
    ClassifierEvalModeName,
    resolve_classifier_eval_mode,
)
from deepeval.config.settings import get_settings
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    initialize_model,
    initialize_system_one_model,
)
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.test_run.api import Classification
from deepeval.utils import get_or_create_event_loop

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


@dataclass(frozen=True)
class ClassificationPrompt:
    """A classify prompt with the content it shows the judge: ``subject``
    names what that content is, ``content`` is its text, and ``shown`` is its
    structured form when it has one (turns or a trace)."""

    text: str
    subject: str
    content: str
    shown: Any = None


class Classifier(BaseClassifier):
    review_schema: Type[LabelReview] = LabelReview

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
        *,
        description: Optional[str] = None,
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
                schema = self._choose_output_schema()
                if schema is not None:
                    result = generate_with_schema_and_extract(
                        metric=self,
                        prompt=prompt.text,
                        schema_cls=schema,
                        extract_schema=lambda result: result,
                        extract_json=schema.model_validate,
                    )
                    merges = merge_new_labels(self, result)
                    self._postprocess_classifications(
                        result, test_case, merges, prompt.shown
                    )
                else:
                    raw_label, reason = generate_classification(
                        self,
                        prompt.text,
                        schema_cls=ClassificationResult,
                        system_one=self._experimental_system_one_spec(
                            test_case
                        ),
                    )
                    self._postprocess_classification(raw_label, reason)
                if self.review_config:
                    review_classifications(self, prompt)

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
            schema = self._choose_output_schema()
            if schema is not None:
                result = await a_generate_with_schema_and_extract(
                    metric=self,
                    prompt=prompt.text,
                    schema_cls=schema,
                    extract_schema=lambda result: result,
                    extract_json=schema.model_validate,
                )
                merges = await a_merge_new_labels(self, result)
                self._postprocess_classifications(
                    result, test_case, merges, prompt.shown
                )
            else:
                raw_label, reason = await a_generate_classification(
                    self,
                    prompt.text,
                    schema_cls=ClassificationResult,
                    system_one=self._experimental_system_one_spec(test_case),
                )
                self._postprocess_classification(raw_label, reason)
            if self.review_config:
                await a_review_classifications(self, prompt)
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
    # Post-processing
    ###############################################

    def _postprocess_classifications(
        self,
        result: Any,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        merges: Dict[str, Label],
        shown: Any,
    ) -> None:
        items = result.labels if self.multi_label else [result]
        none_reason = None
        unresolved = []

        for item in items:
            reason = clean_reason(item.reason) if item.reason else item.reason
            resolved = self._resolve_item(item)
            if resolved.is_none:
                none_reason = none_reason or reason
                continue
            if resolved.unresolved:
                unresolved.append(item.label)
                continue
            label, description = resolved.name, resolved.description
            if label in merges:
                match = merges[label]
                declared = any(l.name == match.name for l in self.labels)
                label = match.name
                description = None if declared else match.description
            elif (
                resolved.origin is LabelOrigin.NEW
                and same_name(label, self.label_pool.labels) is None
            ):
                # Over the max_new_labels cap, so it was never merged or
                # added to the pool.
                continue
            evidence = (
                verify_evidence(
                    getattr(item, "evidence", []),
                    test_case,
                    shown,
                    self.evidence_config.max_quote_chars,
                )
                if self.evidence_config
                else []
            )
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
                    reason=reason if self.include_reason else None,
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

    def _postprocess_classification(
        self, raw_label: Optional[str], reason: Optional[str]
    ) -> None:
        reason = clean_reason(reason) if reason else reason
        # Only reached without label discovery (see _choose_output_schema),
        # so the judge can only answer with a declared label or NONE.
        label = resolve_label(raw_label, self.labels, self.allow_none)
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

    def _resolve_item(self, item: Any) -> ResolvedLabel:
        """Resolve one structured judge answer (a result or a list entry)."""
        reason = clean_reason(item.reason) if item.reason else item.reason
        if self.label_discovery_config:
            return resolve_suggested_label(
                item.label,
                description=getattr(item, "new_label_description", None),
                reason=reason,
                labels=self.labels,
                candidate_labels=self.candidate_labels,
                allow_none=self.allow_none,
            )
        label = resolve_label(item.label, self.labels, self.allow_none)
        if label is None:
            return ResolvedLabel(None)
        if label == NONE_LABEL:
            return ResolvedLabel(NONE_LABEL, LabelOrigin.NONE)
        return ResolvedLabel(label, LabelOrigin.DECLARED)

    def _unresolved_error(self, raw_label: Optional[str]) -> str:
        if raw_label is None:
            return "Judge returned no usable label."
        labels = self.labels
        if self.label_discovery_config:
            labels = labels + self.candidate_labels
        allowed = ", ".join(str(label) for label in labels)
        if self.allow_none:
            allowed += ", or no classification"
        return (
            f"Judge returned '{raw_label}', which is not one of the allowed labels "
            f"({allowed})."
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

    ###############################################
    # Label Discovery
    ###############################################

    @property
    def candidate_labels(self) -> List[Label]:
        return self.label_pool.labels

    ###############################################
    # Prompt
    ###############################################

    def _build_prompt(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> ClassificationPrompt:
        context = self.classification_context
        shared_variables = {
            "labels": format_labels(self.labels),
            "allow_none": self.allow_none,
            "multi_label": self.multi_label,
            "context": self._get_prompt(
                "classification_context",
                name=self.name,
                description=self.description,
                has_context=context is not None,
                app_description=context.app_description if context else None,
                trace_description=(
                    context.trace_description if context else None
                ),
            ),
            "reason_rules": self._get_prompt("reason_rules"),
            "guidance": self._get_prompt("guidance"),
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
            shared_variables["cite_evidence"] = self._cite_evidence(
                "conversation",
                _EVIDENCE_TURN_FIELDS
                + [
                    param.value
                    for param in MULTI_TURN_CONTENT_PARAMS
                    if getattr(test_case, param.value, None)
                ],
            )
            content = construct_multi_turn_content(test_case)
            turns = fit_turns(
                construct_turns(test_case), self.max_content_chars
            )
            return ClassificationPrompt(
                text=self._get_prompt(
                    "classify_multi_turn",
                    test_case_content=content,
                    turns=turns,
                    **shared_variables,
                ),
                subject="conversation",
                content=f"{content}\n\nTurns:\n{turns}",
                shown=turns,
            )
        if test_case._trace_dict is not None:
            shared_variables["cite_evidence"] = self._cite_evidence(
                "trace", _EVIDENCE_TRACE_FIELDS
            )
            trace = fit_trace(
                construct_trace_dict(test_case), self.max_content_chars
            )
            trace_content = construct_trace_content(trace)
            return ClassificationPrompt(
                text=self._get_prompt(
                    "classify_trace",
                    trace_content=trace_content,
                    **shared_variables,
                ),
                subject="trace",
                content=trace_content,
                shown=trace,
            )
        shared_variables["cite_evidence"] = self._cite_evidence(
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
        return ClassificationPrompt(
            text=self._get_prompt(
                "classify_single_turn",
                test_case_content=test_case_content,
                **shared_variables,
            ),
            subject="test case",
            content=test_case_content,
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

    def _choose_output_schema(self) -> Optional[Type[BaseModel]]:
        item = _RESULT_SCHEMAS.get(
            (
                self.label_discovery_config is not None,
                self.evidence_config is not None,
            )
        )
        if self.multi_label:
            return LABEL_LIST_SCHEMAS[item or ClassificationResult]
        return item

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
