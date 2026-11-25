from enum import Enum


class RobustnessType(Enum):
    INPUT_OVERRELIANCE = "input_overreliance"
    HIJACKING = "hijacking"
