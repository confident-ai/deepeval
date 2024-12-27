from enum import Enum


class MathQATask(Enum):
    PROBABILITY = "probability"
    GEOMETRY = "geometry"
    PHYSICS = "physics"
    GAIN = "gain"
    GENERAL = "general"
    OTHER = "other"
