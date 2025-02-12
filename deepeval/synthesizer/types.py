from enum import Enum


class UseCase(Enum):
    QA = "QA"
    TEXT2SQL = "Text to SQL"


class Evolution(Enum):
    REASONING = "Reasoning"
    MULTICONTEXT = "Multi-context"
    CONCRETIZING = "Concretizing"
    CONSTRAINED = "Constrained"
    COMPARATIVE = "Comparative"
    HYPOTHETICAL = "Hypothetical"
    IN_BREADTH = "In-Breadth"


class PromptEvolution(Enum):
    REASONING = "Reasoning"
    CONCRETIZING = "Concretizing"
    CONSTRAINED = "Constrained"
    COMPARATIVE = "Comparative"
    HYPOTHETICAL = "Hypothetical"
    IN_BREADTH = "In-Breadth"
