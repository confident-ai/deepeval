"""Shared helpers for deepeval metrics.

Import from here (``from deepeval.metrics.utils import ...``); the submodules
are an implementation detail:

- ``qag``: the QAG (question-answer generation) verdict loop —
  ``generate_qag_verdicts`` / ``score_qag_verdicts``.
- ``decision``: non-QAG decision points (DAG judgements, G-Eval score) and
  the System One plumbing shared by every Jev call.
- ``system_one``: the whole-chain System One path (`system_one` eval mode).
- ``generation``: generic schema-constrained LLM calls and JSON parsing.
- ``models``: evaluation model / embedding model resolution.
- ``test_case``: test-case parameter validation.
- ``turns``: multi-turn helpers (sliding windows, unit interactions).
- ``verbose``: verbose-log formatting.
- ``metrics``: helpers that operate on lists of metrics.
"""

from .generation import (
    SchemaType,
    ReturnType,
    trimAndLoadJson,
    accrue_token_usage,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from .decision import (
    SystemOneBinarySpec,
    SystemOneChoiceSpec,
    SystemOneScoreSpec,
    system_one_probability,
    a_system_one_probability,
    system_one_score,
    a_system_one_score,
    format_decision_reason,
    has_whole_metric_form,
    generate_binary_judgement,
    a_generate_binary_judgement,
    generate_choice_judgement,
    a_generate_choice_judgement,
    generate_rubric_score,
    a_generate_rubric_score,
    effective_eval_mode,
    reset_system_one_state,
)
from .system_one import (
    SystemOneEvalSpec,
    compact_trace,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    format_system_one_reason,
    format_classification_reason,
)
from .metrics import (
    warn_score_direction_flipped,
    check_at_least_one_metric_has_threshold,
    copy_metrics,
)
from .models import (
    MULTIMODAL_SUPPORTED_MODELS,
    should_use_anthropic_model,
    should_use_azure_openai,
    should_use_local_model,
    should_use_ollama_model,
    should_use_gemini_model,
    should_use_openai_model,
    should_use_litellm,
    should_use_portkey,
    should_use_deepseek_model,
    should_use_openrouter_model,
    should_use_moonshot_model,
    should_use_grok_model,
    should_use_amazon_bedrock_model,
    initialize_model,
    is_native_model,
    initialize_system_one_model,
    should_use_azure_openai_embedding,
    should_use_local_embedding,
    should_use_ollama_embedding,
    initialize_embedding_model,
)
from .qag import (
    SystemOneVerdictSpec,
    split_sentences,
    verdict_from_probability,
    normalize_qag_verdict,
    verdict_from_json,
    score_qag_verdicts,
    generate_qag_verdicts,
    a_generate_qag_verdicts,
    generate_qag_verdict,
    a_generate_qag_verdict,
)
from .test_case import (
    check_conversational_test_case_params,
    check_llm_test_case_params,
    check_arena_test_case_params,
)
from .turns import (
    format_turns,
    convert_turn_to_dict,
    get_turns_in_sliding_window,
    get_unit_interactions,
)
from .verbose import (
    print_tools_called,
    print_verbose_logs,
    construct_verbose_logs,
)

__all__ = [
    # generation
    "SchemaType",
    "ReturnType",
    "trimAndLoadJson",
    "accrue_token_usage",
    "generate_with_schema_and_extract",
    "a_generate_with_schema_and_extract",
    # decision
    "SystemOneBinarySpec",
    "SystemOneChoiceSpec",
    "SystemOneScoreSpec",
    "system_one_probability",
    "a_system_one_probability",
    "system_one_score",
    "a_system_one_score",
    "format_decision_reason",
    "has_whole_metric_form",
    "generate_binary_judgement",
    "a_generate_binary_judgement",
    "generate_choice_judgement",
    "a_generate_choice_judgement",
    "generate_rubric_score",
    "a_generate_rubric_score",
    "effective_eval_mode",
    "reset_system_one_state",
    # system_one
    "SystemOneEvalSpec",
    "compact_trace",
    "parse_questions",
    "run_system_one_eval",
    "a_run_system_one_eval",
    "format_system_one_reason",
    "format_classification_reason",
    # metrics
    "warn_score_direction_flipped",
    "check_at_least_one_metric_has_threshold",
    "copy_metrics",
    # models
    "MULTIMODAL_SUPPORTED_MODELS",
    "should_use_anthropic_model",
    "should_use_azure_openai",
    "should_use_local_model",
    "should_use_ollama_model",
    "should_use_gemini_model",
    "should_use_openai_model",
    "should_use_litellm",
    "should_use_portkey",
    "should_use_deepseek_model",
    "should_use_openrouter_model",
    "should_use_moonshot_model",
    "should_use_grok_model",
    "should_use_amazon_bedrock_model",
    "initialize_model",
    "is_native_model",
    "initialize_system_one_model",
    "should_use_azure_openai_embedding",
    "should_use_local_embedding",
    "should_use_ollama_embedding",
    "initialize_embedding_model",
    # qag
    "SystemOneVerdictSpec",
    "split_sentences",
    "verdict_from_probability",
    "normalize_qag_verdict",
    "verdict_from_json",
    "score_qag_verdicts",
    "generate_qag_verdicts",
    "a_generate_qag_verdicts",
    "generate_qag_verdict",
    "a_generate_qag_verdict",
    # test_case
    "check_conversational_test_case_params",
    "check_llm_test_case_params",
    "check_arena_test_case_params",
    # turns
    "format_turns",
    "convert_turn_to_dict",
    "get_turns_in_sliding_window",
    "get_unit_interactions",
    # verbose
    "print_tools_called",
    "print_verbose_logs",
    "construct_verbose_logs",
]
