from enum import Enum
from typing import Dict

from deepteam.vulnerabilities.types import *


class LLMRiskCategories(Enum):
    RESPONSIBLE_AI = "Responsible AI"
    ILLEGAL = "Illegal"
    BRAND_IMAGE = "Brand Image"
    DATA_PRIVACY = "Data Privacy"
    UNAUTHORIZED_ACCESS = "Unauthorized Access"


def getRiskCategory(
    vulnerability_type: Union[VulnerabilityType, Enum],
) -> LLMRiskCategories:
    risk_category_map: Dict[VulnerabilityType, LLMRiskCategories] = {
        # Responsible AI
        **{bias: LLMRiskCategories.RESPONSIBLE_AI for bias in BiasType},
        **{
            toxicity: LLMRiskCategories.RESPONSIBLE_AI
            for toxicity in ToxicityType
        },
        # Illegal
        **{
            illegal: LLMRiskCategories.ILLEGAL
            for illegal in IllegalActivityType
        },
        **{
            graphic: LLMRiskCategories.ILLEGAL for graphic in GraphicContentType
        },
        **{safety: LLMRiskCategories.ILLEGAL for safety in PersonalSafetyType},
        # Brand Image
        **{
            misinfo: LLMRiskCategories.BRAND_IMAGE
            for misinfo in MisinformationType
        },
        **{
            agency: LLMRiskCategories.BRAND_IMAGE
            for agency in ExcessiveAgencyType
        },
        **{robust: LLMRiskCategories.BRAND_IMAGE for robust in RobustnessType},
        **{
            ip: LLMRiskCategories.BRAND_IMAGE for ip in IntellectualPropertyType
        },
        **{comp: LLMRiskCategories.BRAND_IMAGE for comp in CompetitionType},
        # Data Privacy
        **{
            prompt: LLMRiskCategories.DATA_PRIVACY
            for prompt in PromptLeakageType
        },
        **{pii: LLMRiskCategories.DATA_PRIVACY for pii in PIILeakageType},
        # Unauthorized Access
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS for unauth in BFLAType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS for unauth in BOLAType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS for unauth in RBACType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS
            for unauth in DebugAccessType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS
            for unauth in ShellInjectionType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS
            for unauth in SQLInjectionType
        },
        **{
            unauth: LLMRiskCategories.UNAUTHORIZED_ACCESS for unauth in SSRFType
        },
    }

    return risk_category_map.get(
        vulnerability_type, "Others"
    )  # Returns None if not found
