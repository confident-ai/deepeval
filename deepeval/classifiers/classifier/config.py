from dataclasses import dataclass, field
from typing import List, Optional, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.metrics.utils import initialize_embedding_model
from deepeval.models import DeepEvalBaseEmbeddingModel


@dataclass
class LabelDiscoveryConfig:
    candidate_labels: List[Union[str, Label]] = field(default_factory=list)
    embedding_model: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None
    similarity_threshold: float = 0.92
    top_k: int = 15
    max_new_labels: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.embedding_model, str):
            self.embedding_model = initialize_embedding_model(
                self.embedding_model
            )


@dataclass
class EvidenceConfig:
    max_items: int = 3
    max_quote_chars: int = 300


@dataclass
class ReviewConfig:
    min_confidence: float = 0.6


@dataclass
class ClassificationContext:
    app_description: Optional[str] = None
    trace_description: Optional[str] = None
