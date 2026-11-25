from dataclasses import dataclass, field
from typing import List, Optional, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.metrics.utils import initialize_embedding_model
from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM


@dataclass(frozen=True)
class LabelDiscoveryConfig:
    candidate_labels: List[Union[str, Label]] = field(default_factory=list)
    embedding_model: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None
    similarity_threshold: float = 0.92
    top_k: int = 15
    max_new_labels: Optional[int] = None
    model: Optional[Union[str, DeepEvalBaseLLM]] = None
    min_confidence: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "embedding_model",
            initialize_embedding_model(self.embedding_model),
        )


@dataclass(frozen=True)
class EvidenceConfig:
    max_items: int = 3
    max_quote_chars: int = 300


@dataclass(frozen=True)
class ReviewConfig:
    min_confidence: float = 0.6


@dataclass(frozen=True)
class AgentProfile:
    description: Optional[str] = None
    trace_description: Optional[str] = None
