from typing import Dict, List, Optional
from pydantic import BaseModel
from .aegis.aegis import Aegis
from .frameworks import RedTeamingFramework
from .beavertails.beavertails import BeaverTails
from .owasp.owasp import OWASPTop10
from .nist.nist import NIST
from .mitre.mitre import MITRE
from .risk_category import RiskCategory
from .owasp_top_10_agentic.owasp_top_10_agentic import OWASP_ASI_2026
from .eu_ai_act.eu_ai_act import EUAIAct
from .owasp.risk_categories import OWASP_CATEGORIES
from .nist.risk_categories import NIST_CATEGORIES
from .mitre.risk_categories import MITRE_CATEGORIES
from .owasp_top_10_agentic.risk_categories import OWASP_ASI_CATEGORIES
from .eu_ai_act.risk_categories import EU_AI_ACT_CATEGORIES

FRAMEWORKS_MAP: Dict[str, RedTeamingFramework] = {
    f.name: f for f in [OWASPTop10, NIST, MITRE, OWASP_ASI_2026, EUAIAct]
}

DATASET_FRAMEWORKS_MAP = {f.name: f for f in [Aegis, BeaverTails]}


class RiskCategoryInfo(BaseModel):
    name: str
    vulnerability_types: List[str]
    description: Optional[str] = None
    display_name: Optional[str] = None
    attacks: Optional[List[str]] = None


def _flatten_vulnerability_types(risk_category: RiskCategory) -> List[str]:
    vulnerability_types = []
    for vulnerability in risk_category.vulnerabilities:
        vulnerability_types.extend(vulnerability.ALLOWED_TYPES)
    return vulnerability_types


def _serialize_risk_category(risk_category: RiskCategory) -> RiskCategoryInfo:
    vulnerability_types = _flatten_vulnerability_types(risk_category)
    attack_names = [attack.get_name() for attack in risk_category.attacks]

    return RiskCategoryInfo(
        name=risk_category.name,
        vulnerability_types=vulnerability_types,
        attacks=attack_names,
        description=risk_category.description,
        display_name=risk_category._display_name,
    )


class FrameworkInfo(BaseModel):
    description: str
    risk_categories: List[RiskCategoryInfo]


FRAMEWORK_RISK_CATEGORY_MAPPING: Dict[str, FrameworkInfo] = {
    OWASPTop10.name: FrameworkInfo(
        description=OWASPTop10.description,
        risk_categories=[
            _serialize_risk_category(rc) for rc in OWASP_CATEGORIES
        ],
    ),
    NIST.name: FrameworkInfo(
        description=NIST.description,
        risk_categories=[
            _serialize_risk_category(rc) for rc in NIST_CATEGORIES
        ],
    ),
    MITRE.name: FrameworkInfo(
        description=MITRE.description,
        risk_categories=[
            _serialize_risk_category(rc) for rc in MITRE_CATEGORIES
        ],
    ),
    OWASP_ASI_2026.name: FrameworkInfo(
        description=OWASP_ASI_2026.description,
        risk_categories=[
            _serialize_risk_category(rc) for rc in OWASP_ASI_CATEGORIES
        ],
    ),
    EUAIAct.name: FrameworkInfo(
        description=EUAIAct.description,
        risk_categories=[
            _serialize_risk_category(rc) for rc in EU_AI_ACT_CATEGORIES
        ],
    ),
}
