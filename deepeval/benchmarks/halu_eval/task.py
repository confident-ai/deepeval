from enum import Enum


class HaluEvalTask(Enum):
    QA = "qa"
    DIALOGUE = "dialogue"
    SUMMARIZATION = "summarization"
    GENERAL = "general"
