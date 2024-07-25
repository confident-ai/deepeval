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


class RTAdversarialAttack(Enum):
    PROMPT_INJECTION = "Prompt Injection"
    PROMPT_PROBING = "Prompt Probing"
    GRAY_BOX_ATTACK = "Gray Box Attack"
    JAIL_BREAKING = "Jailbreaking"


class RTVulnerability(Enum):
    HALLUCINATION = "Spread misinformation and hallucinate"
    OFFENSIVE = "Generate harmful content"
    BIAS = "Promote stereotypes and discrimination"
    DATA_LEAKAGE = "Leak confidential data and information"
    UNFORMATTED = "Output undesirable formats"
