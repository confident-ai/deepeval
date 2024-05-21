# Import metric classes from their respective modules
from .answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from .faithfulness.faithfulness import FaithfulnessMetric
from .contextual_recall.contextual_recall import ContextualRecallMetric
from .contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric
from .contextual_precision.contextual_precision import ContextualPrecisionMetric
from .g_eval.g_eval import GEval
from .bias.bias import BiasMetric
from .toxicity.toxicity import ToxicityMetric
from .hallucination.hallucination import HallucinationMetric
from .knowledge_retention.knowledge_retention import KnowledgeRetentionMetric
from .summarization.summarization import SummarizationMetric

# Define a dictionary mapping from metric names to metric classes
metric_class_mapping = {
    'answer_relevancy': AnswerRelevancyMetric,
    'faithfulness': FaithfulnessMetric,
    'contextual_recall': ContextualRecallMetric,
    'contextual_relevancy': ContextualRelevancyMetric,
    'contextual_precision': ContextualPrecisionMetric,
    'geval': GEval,
    'bias': BiasMetric,
    'toxicity': ToxicityMetric,
    'hallucination': HallucinationMetric,
    'knowledge_retention': KnowledgeRetentionMetric,
    'summarization': SummarizationMetric,
}