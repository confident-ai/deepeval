from dataclasses import dataclass, field
from typing import Optional, Union, Dict

from deepeval.metrics.utils import initialize_embedding_model, initialize_model
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.types import Evolution


@dataclass
class FiltrationConfig:
    synthetic_input_quality_threshold: float = 0.5
    max_quality_retries: int = 3
    critic_model: Optional[Union[str, DeepEvalBaseLLM]] = None

    def __post_init__(self):
        self.critic_model, _ = initialize_model(self.critic_model)


@dataclass
class EvolutionConfig:
    num_evolutions: int = 1
    evolutions: Dict[Evolution, float] = field(
        default_factory=lambda: {
            Evolution.REASONING: 1 / 7,
            Evolution.MULTICONTEXT: 1 / 7,
            Evolution.CONCRETIZING: 1 / 7,
            Evolution.CONSTRAINED: 1 / 7,
            Evolution.COMPARATIVE: 1 / 7,
            Evolution.HYPOTHETICAL: 1 / 7,
            Evolution.IN_BREADTH: 1 / 7,
        }
    )


@dataclass
class StylingConfig:
    scenario: Optional[str] = None
    task: Optional[str] = None
    input_format: Optional[str] = None
    expected_output_format: Optional[str] = None


@dataclass
class ConversationalStylingConfig:
    scenario_context: Optional[str] = None
    conversational_task: Optional[str] = None
    participant_roles: Optional[str] = None
    scenario_format: Optional[str] = None
    expected_outcome_format: Optional[str] = None


@dataclass
class ContextConstructionConfig:
    embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None
    critic_model: Optional[Union[str, DeepEvalBaseLLM]] = None
    encoding: Optional[str] = None
    max_contexts_per_document: int = 3
    min_contexts_per_document: int = 1
    max_context_length: int = 3
    min_context_length: int = 1
    chunk_size: int = 1024
    chunk_overlap: int = 0
    context_quality_threshold: float = 0.5
    context_similarity_threshold: float = 0.0
    max_retries: int = 3

    def __post_init__(self):
        self.critic_model, _ = initialize_model(self.critic_model)
        self.embedder = initialize_embedding_model(self.embedder)
