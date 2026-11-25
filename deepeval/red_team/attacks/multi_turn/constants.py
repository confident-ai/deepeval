from typing import Dict
from pydantic import BaseModel

from .base_multi_turn_attack import BaseMultiTurnAttack
from .crescendo_jailbreaking import CrescendoJailbreaking
from .linear_jailbreaking import LinearJailbreaking
from .tree_jailbreaking import TreeJailbreaking
from .sequential_break import SequentialJailbreak
from .bad_likert_judge import BadLikertJudge

MULTI_TURN_ATTACK_CLASSES_MAP: Dict[str, BaseMultiTurnAttack] = {
    v.name: v
    for v in [
        CrescendoJailbreaking,
        LinearJailbreaking,
        TreeJailbreaking,
        SequentialJailbreak,
        BadLikertJudge,
    ]
}

MULTI_TURN_ATTACK_NAMES = sorted(
    v.name for v in MULTI_TURN_ATTACK_CLASSES_MAP.values()
)


class AttackInfo(BaseModel):
    description: str
    exploitability: str


MULTI_TURN_ATTACK_INFO_MAP: Dict[str, AttackInfo] = {
    name: AttackInfo(
        description=attack_class.description,
        exploitability=attack_class.exploitability.value,
    )
    for name, attack_class in MULTI_TURN_ATTACK_CLASSES_MAP.items()
}
