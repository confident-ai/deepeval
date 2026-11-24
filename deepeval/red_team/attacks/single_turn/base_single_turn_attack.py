from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from deepteam.attacks.base_attack import BaseAttack


ParameterType = Literal["string", "integer", "float", "boolean", "enum", "json"]


class AttackParameter(BaseModel):
    type: ParameterType
    required: bool = False
    default: Optional[Any] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None


class BaseSingleTurnAttack(BaseAttack):
    parameters: Dict[str, AttackParameter] = {}
