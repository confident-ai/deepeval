from typing import Dict
from pydantic import BaseModel, Field

from .base_single_turn_attack import BaseSingleTurnAttack, AttackParameter
from .base64 import Base64
from .gray_box import GrayBox
from .leetspeak import Leetspeak
from .math_problem import MathProblem
from .multilingual import Multilingual
from .prompt_injection import PromptInjection
from .prompt_probing import PromptProbing
from .roleplay import Roleplay
from .rot13 import ROT13
from .system_override import SystemOverride
from .adversarial_poetry import AdversarialPoetry
from .character_stream import CharacterStream
from .context_flooding import ContextFlooding
from .embedded_instruction_json import EmbeddedInstructionJSON
from .synthetic_context_injection import SyntheticContextInjection
from .authority_escalation import AuthorityEscalation
from .emotional_manipulation import EmotionalManipulation

from .permission_escalation.permission_escalation import PermissionEscalation
from .goal_redirection.goal_redirection import GoalRedirection
from .semantic_manipulation.semantic_manipulation import LinguisticConfusion
from .input_bypass.input_bypass import InputBypass
from .context_poisoning.context_poisoning import ContextPoisoning

SINGLE_TURN_ATTACK_CLASSES_MAP: Dict[str, BaseSingleTurnAttack] = {
    v.name: v
    for v in [
        AdversarialPoetry,
        Base64,
        CharacterStream,
        ContextFlooding,
        EmbeddedInstructionJSON,
        SyntheticContextInjection,
        AuthorityEscalation,
        EmotionalManipulation,
        GrayBox,
        Leetspeak,
        MathProblem,
        Multilingual,
        PromptInjection,
        PromptProbing,
        Roleplay,
        ROT13,
        SystemOverride,
        PermissionEscalation,
        LinguisticConfusion,
        InputBypass,
        ContextPoisoning,
        GoalRedirection,
    ]
}

SINGLE_TURN_ATTACK_NAMES = sorted(
    v.name for v in SINGLE_TURN_ATTACK_CLASSES_MAP.values()
)


class AttackInfo(BaseModel):
    description: str
    exploitability: str
    parameters: Dict[str, AttackParameter] = Field(default_factory=dict)


# Map attack names to their description and exploitability
SINGLE_TURN_ATTACK_INFO_MAP: Dict[str, AttackInfo] = {
    name: AttackInfo(
        description=attack_class.description,
        exploitability=attack_class.exploitability.value,
        parameters=attack_class.parameters,
    )
    for name, attack_class in SINGLE_TURN_ATTACK_CLASSES_MAP.items()
}
