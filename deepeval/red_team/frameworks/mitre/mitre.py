from typing import List, Literal
from deepteam.frameworks import RedTeamingFramework
from deepteam.frameworks.mitre.risk_categories import MITRE_CATEGORIES

"""
MITRE ATLAS™ (Adversarial Threat Landscape for Artificial-Intelligence Systems)
================================================================================

The MITRE ATLAS™ framework provides a structured knowledge base of adversarial tactics, techniques, 
and procedures (TTPs) used against AI and ML systems. It extends the principles of MITRE ATT&CK® 
to the AI threat surface, identifying how adversaries can manipulate, exploit, or misuse AI models 
throughout their lifecycle.

DeepTeam's MITRE ATLAS module implements these adversarial mappings to test AI and LLM applications 
for security, privacy, and robustness vulnerabilities across each phase of the AI attack lifecycle.

The framework is organized into six adversarial tactics:
1. Reconnaissance (AML.TA0002): Gathering intelligence about AI systems and configurations
2. Resource Development (AML.TA0003): Acquiring resources or tools to enable future attacks
3. Initial Access (AML.TA0004): Gaining entry to the target AI system or environment
4. ML Attack Staging (AML.TA0001): Preparing, training, or adapting attacks specifically for AI models
5. Exfiltration (AML.TA0010): Stealing sensitive information, model data, or internal configurations
6. Impact (AML.TA0011): Manipulating or degrading AI systems to achieve adversarial goals

Each tactic represents a goal an adversary may have while attacking an AI system. These tactics 
collectively map the "why" behind adversarial actions and correspond to different testing modules 
in DeepTeam.

Each category includes:
- Attacks: Realistic adversarial techniques for exploiting, bypassing, or manipulating AI systems
- Vulnerabilities: AI-specific weaknesses that can be leveraged throughout the attack lifecycle

Reference: https://attack.mitre.org
"""


class MITRE(RedTeamingFramework):
    name = "MITRE ATLAS"
    description = "A structured knowledge base of adversarial tactics, techniques, and procedures (TTPs) used against AI and ML systems. Extends MITRE ATT&CK® principles to the AI threat surface, testing across six adversarial tactics: reconnaissance (information gathering), resource development (capability building), initial access (entry point exploitation), ML attack staging (model-specific preparation), exfiltration (data/model theft), and impact (manipulation/degradation)."
    ALLOWED_TYPES = [
        "reconnaissance",
        "resource_development",
        "initial_access",
        "ml_attack_staging",
        "exfiltration",
        "impact",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "reconnaissance",
                "resource_development",
                "initial_access",
                "ml_attack_staging",
                "exfiltration",
                "impact",
            ]
        ] = [
            "reconnaissance",
            "resource_development",
            "initial_access",
            "ml_attack_staging",
            "exfiltration",
            "impact",
        ],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []
        for category in categories:
            for risk_category in MITRE_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
