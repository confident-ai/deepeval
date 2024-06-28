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

class PromptEvolution(Enum):
    REASONING = "Reasoning"
    CONCRETIZING = "Concretizing"
    CONSTRAINED = "Constrained"
    COMPARATIVE = "Comparative"
    HYPOTHETICAL = "Hypothetical"

class RedTeamEvolution(Enum):
    PROMPT_INJECTION = "Prompt Injection"
    PROMPT_PROBING = "Prompt Probing"
    GRAY_BOX_ATTACK = "Gray Box Attack"
    JAIL_BREAKING = "Jailbreaking"

class Response(Enum):
    HALLUCINATION = "spread misinformation and hallucinate"
    OFFENSIVE = "generate harmful content"
    BIAS = "promote stereotypes and discrimination"
    DATA_LEAKAGE = "leak confidential data and information"
    UNFORMATTED = "output undesirable formats"