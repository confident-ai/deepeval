from typing import Dict, Union, Callable
from enum import Enum

from deepeval.vulnerability import BaseVulnerability
from deepeval.vulnerability.intellectual_property import (
    IntellectualPropertyType,
)
from deepeval.vulnerability.unauthorized_access import UnauthorizedAccessType
from deepeval.vulnerability.illegal_activity import IllegalActivityType
from deepeval.vulnerability.excessive_agency import ExcessiveAgencyType
from deepeval.vulnerability.personal_safety import PersonalSafetyType
from deepeval.vulnerability.graphic_content import GraphicContentType
from deepeval.vulnerability.misinformation import MisinformationType
from deepeval.vulnerability.prompt_leakage import PromptLeakageType
from deepeval.vulnerability.competition import CompetitionType
from deepeval.vulnerability.pii_leakage import PIILeakageType
from deepeval.vulnerability.robustness import RobustnessType
from deepeval.vulnerability.toxicity import ToxicityType
from deepeval.vulnerability.bias import BiasType

##########################################
####  Attack Enhancements ################
##########################################


VulnerabilityType = Union[
    UnauthorizedAccessType,
    IllegalActivityType,
    ExcessiveAgencyType,
    PersonalSafetyType,
    GraphicContentType,
    MisinformationType,
    PromptLeakageType,
    PromptLeakageType,
    CompetitionType,
    PIILeakageType,
    RobustnessType,
    ToxicityType,
    BiasType,
    IntellectualPropertyType,
    IntellectualPropertyType,
    IntellectualPropertyType,
]

##########################################
####  Attack Enhancements ################
##########################################


class AttackEnhancement(Enum):
    GRAY_BOX_ATTACK = "Gray Box Attack"
    PROMPT_INJECTION = "Prompt Injection"
    PROMPT_PROBING = "Prompt Probing"
    JAILBREAK_CRESCENDO = "Crescendo Jailbreak"
    JAILBREAK_LINEAR = "Linear Jailbreak"
    JAILBREAK_TREE = "Tree Jailbreak"
    ROT13 = "ROT13 Encoding"
    BASE64 = "Base64 Encoding"
    LEETSPEAK = "Leetspeak Encoding"
    MATH_PROBLEM = "Math Problem"
    MULTILINGUAL = "Multilingual"


##########################################
#### LLM Risk Categories #################
##########################################


class LLMRiskCategories(Enum):
    RESPONSIBLE_AI = "Responsible AI"
    ILLEGAL = "Illegal"
    BRAND_IMAGE = "Brand Image"
    DATA_PRIVACY = "Data Privacy"
    UNAUTHORIZED_ACCESS = "Unauthorized Access"


llm_risk_categories_map: Dict[BaseVulnerability, LLMRiskCategories] = {
    #### Responsible AI ####
    BiasType: LLMRiskCategories.RESPONSIBLE_AI,
    ToxicityType: LLMRiskCategories.RESPONSIBLE_AI,
    #### Illegal ####
    IllegalActivityType: LLMRiskCategories.ILLEGAL,
    GraphicContentType: LLMRiskCategories.ILLEGAL,
    PersonalSafetyType: LLMRiskCategories.ILLEGAL,
    #### Brand Image ####
    MisinformationType: LLMRiskCategories.BRAND_IMAGE,
    ExcessiveAgencyType: LLMRiskCategories.BRAND_IMAGE,
    RobustnessType: LLMRiskCategories.BRAND_IMAGE,
    IntellectualPropertyType: LLMRiskCategories.BRAND_IMAGE,
    CompetitionType: LLMRiskCategories.BRAND_IMAGE,
    #### Data Privacy ####
    PromptLeakageType: LLMRiskCategories.DATA_PRIVACY,
    PIILeakageType: LLMRiskCategories.DATA_PRIVACY,
    #### Unauthorized Access ####
    UnauthorizedAccessType: LLMRiskCategories.UNAUTHORIZED_ACCESS,
}

##########################################
#### LLM Model ###########################
##########################################

CallbackType = Callable[[str], str]
