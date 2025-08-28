import importlib
from typing import TYPE_CHECKING

__all__ = [
    # Base classes
    "BaseMetric",
    "BaseConversationalMetric",
    "BaseMultimodalMetric",
    "BaseArenaMetric",
    # Core metrics
    "GEval",
    "ArenaGEval",
    "ConversationalGEval",
    "DAGMetric",
    # RAG metrics
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    # MCP metrics
    "MCPTaskCompletionMetric",
    "MultiTurnMCPUseMetric",
    "MCPUseMetric",
    # Content quality metrics
    "HallucinationMetric",
    "BiasMetric",
    "ToxicityMetric",
    "SummarizationMetric",
    # Safety and compliance metrics
    "PIILeakageMetric",
    "NonAdviceMetric",
    "MisuseMetric",
    "RoleViolationMetric",
    "RoleAdherenceMetric",
    # Task-specific metrics
    "ToolCorrectnessMetric",
    "JsonCorrectnessMetric",
    "PromptAlignmentMetric",
    "TaskCompletionMetric",
    "ArgumentCorrectnessMetric",
    "KnowledgeRetentionMetric",
    # Conversational metrics
    "TurnRelevancyMetric",
    "ConversationCompletenessMetric",
    # Multimodal metrics
    "TextToImageMetric",
    "ImageEditingMetric",
    "ImageCoherenceMetric",
    "ImageHelpfulnessMetric",
    "ImageReferenceMetric",
    "MultimodalContextualRecallMetric",
    "MultimodalContextualRelevancyMetric",
    "MultimodalContextualPrecisionMetric",
    "MultimodalAnswerRelevancyMetric",
    "MultimodalFaithfulnessMetric",
    "MultimodalToolCorrectnessMetric",
    "MultimodalGEval",
]

_SYMBOL_TO_MODULE = {
    # Base classes
    "BaseMetric": "deepeval.metrics.base_metric",
    "BaseConversationalMetric": "deepeval.metrics.base_metric",
    "BaseMultimodalMetric": "deepeval.metrics.base_metric",
    "BaseArenaMetric": "deepeval.metrics.base_metric",
    # Core metrics
    "GEval": "deepeval.metrics.g_eval.g_eval",
    "ArenaGEval": "deepeval.metrics.arena_g_eval.arena_g_eval",
    "ConversationalGEval": "deepeval.metrics.conversational_g_eval.conversational_g_eval",
    "DAGMetric": "deepeval.metrics.dag.dag",
    # RAG metrics
    "AnswerRelevancyMetric": "deepeval.metrics.answer_relevancy.answer_relevancy",
    "FaithfulnessMetric": "deepeval.metrics.faithfulness.faithfulness",
    "ContextualRecallMetric": "deepeval.metrics.contextual_recall.contextual_recall",
    "ContextualRelevancyMetric": "deepeval.metrics.contextual_relevancy.contextual_relevancy",
    "ContextualPrecisionMetric": "deepeval.metrics.contextual_precision.contextual_precision",
    # MCP metrics
    "MCPTaskCompletionMetric": "deepeval.metrics.mcp.mcp_task_completion",
    "MultiTurnMCPUseMetric": "deepeval.metrics.mcp.multi_turn_mcp_use_metric",
    "MCPUseMetric": "deepeval.metrics.mcp_use_metric.mcp_use_metric",
    # Content quality metrics
    "HallucinationMetric": "deepeval.metrics.hallucination.hallucination",
    "BiasMetric": "deepeval.metrics.bias.bias",
    "ToxicityMetric": "deepeval.metrics.toxicity.toxicity",
    "SummarizationMetric": "deepeval.metrics.summarization.summarization",
    # Safety and compliance metrics
    "PIILeakageMetric": "deepeval.metrics.pii_leakage.pii_leakage",
    "NonAdviceMetric": "deepeval.metrics.non_advice.non_advice",
    "MisuseMetric": "deepeval.metrics.misuse.misuse",
    "RoleViolationMetric": "deepeval.metrics.role_violation.role_violation",
    "RoleAdherenceMetric": "deepeval.metrics.role_adherence.role_adherence",
    # Task-specific metrics
    "ToolCorrectnessMetric": "deepeval.metrics.tool_correctness.tool_correctness",
    "JsonCorrectnessMetric": "deepeval.metrics.json_correctness.json_correctness",
    "PromptAlignmentMetric": "deepeval.metrics.prompt_alignment.prompt_alignment",
    "TaskCompletionMetric": "deepeval.metrics.task_completion.task_completion",
    "ArgumentCorrectnessMetric": "deepeval.metrics.argument_correctness.argument_correctness",
    "KnowledgeRetentionMetric": "deepeval.metrics.knowledge_retention.knowledge_retention",
    # Conversational metrics
    "TurnRelevancyMetric": "deepeval.metrics.turn_relevancy.turn_relevancy",
    "ConversationCompletenessMetric": "deepeval.metrics.conversation_completeness.conversation_completeness",
    # Multimodal metrics
    "TextToImageMetric": "deepeval.metrics.multimodal_metrics.text_to_image.text_to_image",
    "ImageEditingMetric": "deepeval.metrics.multimodal_metrics.image_editing.image_editing",
    "ImageCoherenceMetric": "deepeval.metrics.multimodal_metrics.image_coherence.image_coherence",
    "ImageHelpfulnessMetric": "deepeval.metrics.multimodal_metrics.image_helpfulness.image_helpfulness",
    "ImageReferenceMetric": "deepeval.metrics.multimodal_metrics.image_reference.image_reference",
    "MultimodalContextualRecallMetric": "deepeval.metrics.multimodal_metrics.multimodal_contextual_recall.multimodal_contextual_recall",
    "MultimodalContextualRelevancyMetric": "deepeval.metrics.multimodal_metrics.multimodal_contextual_relevancy.multimodal_contextual_relevancy",
    "MultimodalContextualPrecisionMetric": "deepeval.metrics.multimodal_metrics.multimodal_contextual_precision.multimodal_contextual_precision",
    "MultimodalAnswerRelevancyMetric": "deepeval.metrics.multimodal_metrics.multimodal_answer_relevancy.multimodal_answer_relevancy",
    "MultimodalFaithfulnessMetric": "deepeval.metrics.multimodal_metrics.multimodal_faithfulness.multimodal_faithfulness",
    "MultimodalToolCorrectnessMetric": "deepeval.metrics.multimodal_metrics.multimodal_tool_correctness.multimodal_tool_correctness",
    "MultimodalGEval": "deepeval.metrics.multimodal_metrics.multimodal_g_eval.multimodal_g_eval",
}


def __getattr__(name: str):
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = importlib.import_module(module_path)
    obj = getattr(module, name)
    globals()[name] = obj
    return obj


if TYPE_CHECKING:
    from .answer_relevancy.answer_relevancy import AnswerRelevancyMetric
    from .arena_g_eval.arena_g_eval import ArenaGEval
    from .argument_correctness.argument_correctness import ArgumentCorrectnessMetric
    from .base_metric import (
        BaseArenaMetric,
        BaseConversationalMetric,
        BaseMetric,
        BaseMultimodalMetric,
    )
    from .bias.bias import BiasMetric
    from .contextual_precision.contextual_precision import ContextualPrecisionMetric
    from .contextual_recall.contextual_recall import ContextualRecallMetric
    from .contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric
    from .conversation_completeness.conversation_completeness import (
        ConversationCompletenessMetric,
    )
    from .conversational_g_eval.conversational_g_eval import ConversationalGEval
    from .dag.dag import DAGMetric
    from .faithfulness.faithfulness import FaithfulnessMetric
    from .g_eval.g_eval import GEval
    from .hallucination.hallucination import HallucinationMetric
    from .json_correctness.json_correctness import JsonCorrectnessMetric
    from .knowledge_retention.knowledge_retention import KnowledgeRetentionMetric
    from .mcp.mcp_task_completion import MCPTaskCompletionMetric
    from .mcp.multi_turn_mcp_use_metric import MultiTurnMCPUseMetric
    from .mcp_use_metric.mcp_use_metric import MCPUseMetric
    from .misuse.misuse import MisuseMetric
    from .multimodal_metrics.image_coherence.image_coherence import ImageCoherenceMetric
    from .multimodal_metrics.image_editing.image_editing import ImageEditingMetric
    from .multimodal_metrics.image_helpfulness.image_helpfulness import (
        ImageHelpfulnessMetric,
    )
    from .multimodal_metrics.image_reference.image_reference import ImageReferenceMetric
    from .multimodal_metrics.multimodal_answer_relevancy.multimodal_answer_relevancy import (
        MultimodalAnswerRelevancyMetric,
    )
    from .multimodal_metrics.multimodal_contextual_precision.multimodal_contextual_precision import (
        MultimodalContextualPrecisionMetric,
    )
    from .multimodal_metrics.multimodal_contextual_recall.multimodal_contextual_recall import (
        MultimodalContextualRecallMetric,
    )
    from .multimodal_metrics.multimodal_contextual_relevancy.multimodal_contextual_relevancy import (
        MultimodalContextualRelevancyMetric,
    )
    from .multimodal_metrics.multimodal_faithfulness.multimodal_faithfulness import (
        MultimodalFaithfulnessMetric,
    )
    from .multimodal_metrics.multimodal_g_eval.multimodal_g_eval import MultimodalGEval
    from .multimodal_metrics.multimodal_tool_correctness.multimodal_tool_correctness import (
        MultimodalToolCorrectnessMetric,
    )

    # Multimodal metrics
    from .multimodal_metrics.text_to_image.text_to_image import TextToImageMetric
    from .non_advice.non_advice import NonAdviceMetric
    from .pii_leakage.pii_leakage import PIILeakageMetric
    from .prompt_alignment.prompt_alignment import PromptAlignmentMetric
    from .role_adherence.role_adherence import RoleAdherenceMetric
    from .role_violation.role_violation import RoleViolationMetric
    from .summarization.summarization import SummarizationMetric
    from .task_completion.task_completion import TaskCompletionMetric
    from .tool_correctness.tool_correctness import ToolCorrectnessMetric
    from .toxicity.toxicity import ToxicityMetric
    from .turn_relevancy.turn_relevancy import TurnRelevancyMetric
