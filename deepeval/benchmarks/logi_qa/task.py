from enum import Enum


class LogiQATask(Enum):
    CATEGORICAL_REASONING = "Categorical Reasoning"
    SUFFICIENT_CONDITIONAL_REASONING = "Sufficient Conditional Reasoning"
    NECESSARY_CONDITIONAL_REASONING = "Necessry Condtional Reasoning"
    DISJUNCTIVE_REASONING = "Disjunctive Reasoning"
    CONJUNCTIVE_REASONING = "Conjunctive Reasoning"
