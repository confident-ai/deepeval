from typing import List, Optional
from enum import Enum
from deepeval.test_case import LLMTestCase, Turn, ToolCall


class RTTurn(Turn):
    turn_level_attack: Optional[str] = None
    stopping_reason: Optional[str] = None
    stopping_category: Optional[str] = None

    def __repr__(self):
        attrs = [f"role={self.role}", f"content={self.content}"]
        if self.turn_level_attack is not None:
            attrs.append(f"turn_level_attack={self.turn_level_attack}")
        if self.stopping_category is not None:
            attrs.append(f"stopping_category={self.stopping_category}")
        if self.stopping_reason is not None:
            attrs.append(f"stopping_reason={self.stopping_reason}")
        return f"RTTurn({', '.join(attrs)})"


class RTTestCase(LLMTestCase):
    vulnerability: str
    input: Optional[str] = None
    actual_output: Optional[str] = None
    turns: Optional[List[RTTurn]] = None
    metadata: Optional[dict] = None
    vulnerability_type: Enum = None
    attack_method: Optional[str] = None
    risk_category: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    cvss_score: Optional[float] = None
    simulation_cost: Optional[float] = None
    evaluation_cost: Optional[float] = None

    def __repr__(self):
        attrs = [
            f"vulnerability={self.vulnerability}",
            f"vulnerability_type={self.vulnerability_type.value}",
        ]
        if self.input is not None:
            attrs.append(f"input={self.input}")
        if self.actual_output is not None:
            attrs.append(f"actual_output={self.actual_output}")
        if self.turns is not None:
            attrs.append(f"turns={self.turns}")
        if self.metadata is not None:
            attrs.append(f"metadata={self.metadata}")
        if self.attack_method is not None:
            attrs.append(f"attack_method={self.attack_method}")
        if self.risk_category is not None:
            attrs.append(f"risk_category={self.risk_category}")
        if self.score is not None:
            attrs.append(f"score={self.score}")
        if self.reason is not None:
            attrs.append(f"reason={self.reason}")
        if self.error is not None:
            attrs.append(f"error={self.error}")
        return f"RTTestCase({', '.join(attrs)})"

    # @model_validator(mode="before")
    # def validate_input(cls, data):
    #     vulnerability = data.get("vulnerability")
    #     input = data.get("input")
    #     actual_output = data.get("actual_output")
    #     turns = data.get("turns")
    #     vulnerability_type = data.get("vulnerability_type")
    #     metadata = data.get("metadata")
    #     attack_method = data.get("attack_method")
    #     risk_category = data.get("risk_category")
    #     score = data.get("score")
    #     reason = data.get("reason")
    #     error = data.get("error")

    #     if input is not None:
    #         if not isinstance(input, str):
    #             raise TypeError("'input' must be a string")

    #     if actual_output is not None:
    #         if not isinstance(actual_output, str):
    #             raise TypeError("'actual_output' must be a string")

    #     if turns is not None:
    #         if not isinstance(turns, list) or not all(
    #             isinstance(turn, RTTurn) for turn in turns
    #         ):
    #             raise TypeError("'turns' must be a list of 'RTTurn'")

    #     if actual_output is not None and turns is not None:
    #         raise ValueError(
    #             "An 'RTTestCase' cannot contain both 'actual_output' and 'turns' at the same time."
    #         )

    #     if vulnerability is not None:
    #         if not isinstance(vulnerability, str):
    #             raise TypeError("'vulnerability' must be a string")

    #     if vulnerability_type is not None:
    #         if not isinstance(vulnerability, str):
    #             raise TypeError("'vulnerability_type' must be an Enum")

    #     if metadata is not None:
    #         if not isinstance(metadata, dict):
    #             raise TypeError("'metadata' must be a dictionary")

    #     if attack_method is not None:
    #         if not isinstance(attack_method, str):
    #             raise TypeError("'attack_method' must be a string")

    #     if risk_category is not None:
    #         if not isinstance(risk_category, str):
    #             raise TypeError("'risk_category' must be a string")

    #     if score is not None:
    #         if not isinstance(score, float):
    #             raise TypeError("'score' must be a float")

    #     if reason is not None:
    #         if not isinstance(reason, str):
    #             raise TypeError("'reason' must be a string")

    #     if error is not None:
    #         if not isinstance(error, str):
    #             raise TypeError("'error' must be a string")

    #     return data
