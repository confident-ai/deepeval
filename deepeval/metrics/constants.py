from .bias.bias import BiasMetric
from .exact_match.exact_match import ExactMatchMetric
from .pattern_match.pattern_match import PatternMatchMetric
from .toxicity.toxicity import ToxicityMetric
from .pii_leakage.pii_leakage import PIILeakageMetric
from .non_advice.non_advice import NonAdviceMetric
from .misuse.misuse import MisuseMetric
from .hallucination.hallucination import HallucinationMetric
from .answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from .summarization.summarization import SummarizationMetric
from .g_eval.g_eval import GEval
from .faithfulness.faithfulness import FaithfulnessMetric
from .contextual_recall.contextual_recall import ContextualRecallMetric
from .contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric
from .contextual_precision.contextual_precision import ContextualPrecisionMetric
from .knowledge_retention.knowledge_retention import KnowledgeRetentionMetric
from .tool_correctness.tool_correctness import ToolCorrectnessMetric
from .json_correctness.json_correctness import JsonCorrectnessMetric
from .prompt_alignment.prompt_alignment import PromptAlignmentMetric
from .task_completion.task_completion import TaskCompletionMetric
from .topic_adherence.topic_adherence import TopicAdherenceMetric
from .step_efficiency.step_efficiency import StepEfficiencyMetric
from .plan_adherence.plan_adherence import PlanAdherenceMetric
from .plan_quality.plan_quality import PlanQualityMetric
from .tool_use.tool_use import ToolUseMetric
from .goal_accuracy.goal_accuracy import GoalAccuracyMetric
from .argument_correctness.argument_correctness import ArgumentCorrectnessMetric
from .turn_relevancy.turn_relevancy import (
    TurnRelevancyMetric,
)
from .turn_faithfulness.turn_faithfulness import TurnFaithfulnessMetric
from .turn_contextual_precision.turn_contextual_precision import (
    TurnContextualPrecisionMetric,
)
from .turn_contextual_recall.turn_contextual_recall import (
    TurnContextualRecallMetric,
)
from .turn_contextual_relevancy.turn_contextual_relevancy import (
    TurnContextualRelevancyMetric,
)
from .conversation_completeness.conversation_completeness import (
    ConversationCompletenessMetric,
)
from .role_adherence.role_adherence import (
    RoleAdherenceMetric,
)
from .conversational_g_eval.conversational_g_eval import ConversationalGEval
from .multimodal_metrics import (
    TextToImageMetric,
    ImageEditingMetric,
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
)


SINGLE_TURN_METRICS = [
    ExactMatchMetric,
    PatternMatchMetric,
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    SummarizationMetric,
    PIILeakageMetric,
    NonAdviceMetric,
    MisuseMetric,
    ToolCorrectnessMetric,
    JsonCorrectnessMetric,
    PromptAlignmentMetric,
    TaskCompletionMetric,
    ArgumentCorrectnessMetric,
    StepEfficiencyMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    TextToImageMetric,
    ImageEditingMetric,
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
]

MULTI_TURN_METRICS = [
    ConversationalGEval,
    TopicAdherenceMetric,
    RoleAdherenceMetric,
    ConversationCompletenessMetric,
    TurnRelevancyMetric,
    TurnFaithfulnessMetric,
    GoalAccuracyMetric,
    KnowledgeRetentionMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    ToolUseMetric,
]

SINGLE_TURN_METRICS_MAPPING = {
    metric.__name__: metric for metric in SINGLE_TURN_METRICS
}

MULTI_TURN_METRICS_MAPPING = {
    metric.__name__: metric for metric in MULTI_TURN_METRICS
}
