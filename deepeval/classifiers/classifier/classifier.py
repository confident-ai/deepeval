"""LLM-judged classifier: assigns one or more labels to a test case."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type, Union, cast

from deepeval.classifiers.base_classifier import (
    NONE_LABEL,
    BaseClassifier,
    Label,
)
from deepeval.classifiers.classifier.config import (
    AgentProfile,
    EvidenceConfig,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.classifiers.classifier.schema import (
    LABEL_LIST_SCHEMAS,
    ClassificationResult,
    ClassificationResults,
    EvidencedClassificationResult,
    LabelReview,
    SuggestedClassificationResult,
    SuggestedEvidencedClassificationResult,
)
from deepeval.classifiers.classifier.steps.classification import (
    a_generate_classification_results,
    generate_classification_results,
)
from deepeval.classifiers.classifier.steps.evidence import (
    TRUNCATION_MARKER_REASON,
    blames_truncation_marker,
    citable_fields,
    verify_evidence,
)
from deepeval.classifiers.classifier.steps.label_discovery import (
    LabelPool,
    MergedLabels,
    a_merge_new_labels,
    merge_new_labels,
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
    TRUNCATION_MARKER,
    ShownContent,
    fit_text,
    fit_trace,
    fit_turns,
)
from deepeval.classifiers.utils import (
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
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.test_run.api import Classification
from deepeval.utils import get_or_create_event_loop

ClassifierTemplate = make_template_class("Classifier", feature="classifiers")

# Built-in classifiers (``RefusalClassifier`` ...) share ``Classifier``'s bundle
# entry. Public templates reach it through ``classification_template``; the
# ``_experimental_*`` ones are not exposed on the template class, so they name
# the bundle key explicitly.
_EXPERIMENTAL_TEMPLATE_CLASS = "Classifier"


@dataclass(frozen=True)
class ClassificationPrompt:
    """A classify prompt with the content it shows the judge: ``subject``
    names what that content is, ``content`` is its text, ``shown`` is its
    structured form when it has one (turns or a trace), and
    ``evidence_fields`` are the fields the judge may quote as evidence."""

    text: str
    subject: str
    content: str
    evidence_fields: List[str]
    shown: Optional[ShownContent] = None


class Classifier(BaseClassifier):
    """Labels a test case, conversation or trace with an LLM judge, choosing
    from your labels and, when configured, citing evidence, reviewing each
    label and discovering new ones."""

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
        agent_profile: Optional[AgentProfile] = None,
    ):
        if not name or not name.strip():
            raise ValueError("Classifier 'name' cannot be empty.")

        self.name = name
        self.labels = (
            normalize_labels(labels)
            if labels or label_discovery_config is None
            else []
        )
        self.eval_mode = resolve_classifier_eval_mode(eval_mode)
        if self.eval_mode.uses_system_one and (
            multi_label
            or evidence_config is not None
            or review_config is not None
            or label_discovery_config is not None
        ):
            raise ValueError(
                "multi_label, evidence_config, review_config and label_discovery_config need an LLM judge; they are not supported in the system_one eval mode."
            )
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
        self.description = description
        self.multi_label = multi_label
        if max_content_chars <= len(TRUNCATION_MARKER):
            raise ValueError(
                f"max_content_chars must be greater than {len(TRUNCATION_MARKER)}."
            )
        self.max_content_chars = max_content_chars
        self.evidence_config = evidence_config
        self.review_config = review_config
        self.label_discovery_config = label_discovery_config
        self.agent_profile = agent_profile

    @property
    def candidate_labels(self) -> List[Label]:
        return self.label_pool.labels

    ###############################################
    # Classifications
    ###############################################

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
                    classification_results = generate_classification_results(
                        self, prompt.text, schema
                    )
                    merged_labels = merge_new_labels(
                        self, classification_results
                    )
                    self._postprocess_classifications(
                        classification_results, test_case, prompt, merged_labels
                    )
                else:
                    raw_label, reason = generate_classification(
                        self,
                        prompt.text,
                        schema_cls=ClassificationResult,
                        system_one=(
                            self._experimental_system_one_spec(test_case)
                            if self.eval_mode.uses_system_one
                            else None
                        ),
                    )
                    self._postprocess_classifications(
                        [
                            ClassificationResult.model_construct(
                                label=raw_label, reason=reason
                            )
                        ],
                        test_case,
                        prompt,
                        MergedLabels(),
                    )
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
                classification_results = (
                    await a_generate_classification_results(
                        self, prompt.text, schema
                    )
                )
                merged_labels = await a_merge_new_labels(
                    self, classification_results
                )
                self._postprocess_classifications(
                    classification_results, test_case, prompt, merged_labels
                )
            else:
                raw_label, reason = await a_generate_classification(
                    self,
                    prompt.text,
                    schema_cls=ClassificationResult,
                    system_one=(
                        self._experimental_system_one_spec(test_case)
                        if self.eval_mode.uses_system_one
                        else None
                    ),
                )
                self._postprocess_classifications(
                    [
                        ClassificationResult.model_construct(
                            label=raw_label, reason=reason
                        )
                    ],
                    test_case,
                    prompt,
                    MergedLabels(),
                )
            if self.review_config:
                await a_review_classifications(self, prompt)
            return self.label

    def copy(self) -> "Classifier":
        clone = cast("Classifier", super().copy())
        clone.label_pool = self.label_pool
        return clone

    ###############################################
    # Post-process classifications
    ###############################################

    def _postprocess_classifications(
        self,
        classification_results: List[ClassificationResult],
        test_case: Union[LLMTestCase, ConversationalTestCase],
        prompt: ClassificationPrompt,
        merged_labels: MergedLabels,
    ) -> None:
        no_label_reason = None
        unresolved_labels = []

        for classification_result in classification_results:
            reason = clean_reason(classification_result.reason)
            resolved_label = self._resolve_label_from_classification_result(
                classification_result, reason
            )

            # No resolved label
            if resolved_label.is_none:
                no_label_reason = no_label_reason or reason
                continue
            if resolved_label.unresolved:
                unresolved_labels.append(classification_result.label)
                continue

            # Merged label
            if resolved_label.origin is LabelOrigin.NEW:
                merged_label = merged_labels.get(resolved_label.name)
                if merged_label is None:
                    no_label_reason = no_label_reason or (
                        f'Proposed new label "{resolved_label.name}", but '
                        "max_new_labels was already reached."
                    )
                    continue
                resolved_label = merged_label
            label, description = resolved_label.name, resolved_label.description

            # Truncation marker
            cited = (
                classification_result.evidence
                if isinstance(
                    classification_result, EvidencedClassificationResult
                )
                else []
            )
            if blames_truncation_marker(label, description, reason, cited):
                no_label_reason = no_label_reason or TRUNCATION_MARKER_REASON
                continue

            # Evidence
            evidence = (
                verify_evidence(cited, test_case, prompt, self.evidence_config)
                if self.evidence_config
                else []
            )

            # Duplicate label
            existing_classification = next(
                (
                    classification
                    for classification in self.classifications
                    if classification.label == label
                ),
                None,
            )
            if existing_classification is not None:
                if self.evidence_config:
                    combined = {
                        (e.field, e.quote, e.turn_index, e.span_index): e
                        for e in existing_classification.evidence + evidence
                    }
                    existing_classification.evidence = list(combined.values())[
                        : self.evidence_config.max_items
                    ]
                continue

            # New classification
            self.classifications.append(
                Classification(
                    name=self.name,
                    label=label,
                    reason=reason if self.include_reason else None,
                    new_label_description=description,
                    evidence=evidence,
                )
            )

        # Top classification
        top_classification = (
            self.classifications[0] if self.classifications else None
        )
        self.label = top_classification.label if top_classification else None
        top_reason = (
            top_classification.reason if top_classification else no_label_reason
        )
        self.reason = top_reason if self.include_reason else None

        # No classification
        if top_classification is None and (
            unresolved_labels or not self.allow_none
        ):
            self.error = self._unresolved_error(
                unresolved_labels[0] if unresolved_labels else None
            )

    def _resolve_label_from_classification_result(
        self, classification_result: ClassificationResult, reason: str
    ) -> ResolvedLabel:
        # Label discovery: a declared, candidate or new label
        if self.label_discovery_config:
            return resolve_suggested_label(
                classification_result.label,
                description=(
                    classification_result.new_label_description
                    if isinstance(
                        classification_result, SuggestedClassificationResult
                    )
                    else None
                ),
                reason=reason,
                labels=self.labels,
                candidate_labels=self.candidate_labels,
                allow_none=self.allow_none,
                classifier_name=self.name,
            )

        # Declared labels only
        label = resolve_label(
            classification_result.label, self.labels, self.allow_none
        )
        if label is None:
            return ResolvedLabel(None)
        origin = (
            LabelOrigin.NONE if label == NONE_LABEL else LabelOrigin.DECLARED
        )
        return ResolvedLabel(label, origin)

    def _unresolved_error(self, judge_label: Optional[str]) -> str:
        if judge_label is None:
            return "Judge returned no usable label."
        allowed_labels = self.labels + (
            self.candidate_labels if self.label_discovery_config else []
        )
        allowed = ", ".join(label.name for label in allowed_labels)
        if self.allow_none:
            allowed += ", or no classification"
        return (
            f"Judge returned '{judge_label}', which is not one of the "
            f"allowed labels ({allowed})."
        )

    ###############################################
    # Prompt
    ###############################################

    def _build_prompt(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> ClassificationPrompt:
        shown: Optional[ShownContent] = None

        # Conversation
        if isinstance(test_case, ConversationalTestCase):
            test_case_content = fit_text(
                construct_multi_turn_content(test_case),
                self.max_content_chars // 2,
            )
            shown = fit_turns(
                [
                    {"index": index, **turn}
                    for index, turn in enumerate(construct_turns(test_case), 1)
                ],
                self.max_content_chars - len(test_case_content),
            )
            template, subject = "classify_multi_turn", "conversation"
            content = f"{test_case_content}\n\nTurns:\n{shown}"
            content_variables = {
                "test_case_content": test_case_content,
                "turns": shown,
            }

        # Trace
        elif test_case._trace_dict is not None:
            shown = fit_trace(
                construct_trace_dict(test_case), self.max_content_chars
            )
            template, subject = "classify_traced_single_turn", "trace"
            content = construct_trace_content(shown)
            content_variables = {"trace_content": content}

        # Single-turn test case
        else:
            template, subject = "classify_single_turn", "test case"
            content = fit_text(
                construct_single_turn_content(test_case),
                self.max_content_chars,
            )
            content_variables = {"test_case_content": content}

        # Prompt
        evidence_fields = citable_fields(test_case)
        return ClassificationPrompt(
            text=self._get_prompt(
                template,
                **content_variables,
                **self._shared_prompt_variables(subject, evidence_fields),
            ),
            subject=subject,
            content=content,
            evidence_fields=evidence_fields,
            shown=shown,
        )

    def _shared_prompt_variables(
        self, subject: str, evidence_fields: List[str]
    ) -> Dict[str, Union[str, bool]]:
        agent_profile = self.agent_profile
        return {
            "labels": format_labels(self.labels),
            "allow_none": self.allow_none,
            "multi_label": self.multi_label,
            "context": self._get_prompt(
                "agent_profile",
                name=self.name,
                description=self.description,
                has_context=agent_profile is not None,
                app_description=(
                    agent_profile.description if agent_profile else None
                ),
                trace_description=(
                    agent_profile.trace_description if agent_profile else None
                ),
            ),
            "reason_rules": self._get_prompt("reason_rules"),
            "guidance": self._get_prompt("guidance"),
            "cite_evidence": (
                self._get_prompt(
                    "cite_evidence",
                    subject=subject,
                    fields=", ".join(f'"{field}"' for field in evidence_fields),
                    allow_none=self.allow_none,
                    max_items=self.evidence_config.max_items,
                    max_quote_chars=self.evidence_config.max_quote_chars,
                )
                if self.evidence_config
                else ""
            ),
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

    def _experimental_system_one_spec(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> SystemOneClassifySpec:
        """Used when the eval mode is `system_one`; see EXPERIMENTAL.md.

        The label is one Choice over the declared labels (plus ``NONE`` when
        ``allow_none``), and the reason is built from the answer. A test case
        with a trace is shown as the same trace the LLM judge sees.
        """
        multi_turn = isinstance(test_case, ConversationalTestCase)
        traced = not multi_turn and test_case._trace_dict is not None
        if multi_turn:
            state = construct_multi_turn_state(test_case)
            state["turns"] = fit_turns(state["turns"], self.max_content_chars)
        elif traced:
            state = {
                "trace": fit_trace(
                    construct_trace_dict(test_case), self.max_content_chars
                )
            }
        else:
            state = construct_single_turn_state(test_case)
            state["test_case"] = {
                field: (
                    fit_text(value, self.max_content_chars)
                    if isinstance(value, str)
                    else value
                )
                for field, value in state["test_case"].items()
            }
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
                traced=traced,
            ),
            options=options,
            state=state,
            include_reason=self.include_reason,
        )

    def _choose_output_schema(
        self,
    ) -> Optional[Type[Union[ClassificationResult, ClassificationResults]]]:
        if self.label_discovery_config and self.evidence_config:
            schema = SuggestedEvidencedClassificationResult
        elif self.label_discovery_config:
            schema = SuggestedClassificationResult
        elif self.evidence_config:
            schema = EvidencedClassificationResult
        elif self.multi_label:
            schema = ClassificationResult
        else:
            return None
        return LABEL_LIST_SCHEMAS[schema] if self.multi_label else schema

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
