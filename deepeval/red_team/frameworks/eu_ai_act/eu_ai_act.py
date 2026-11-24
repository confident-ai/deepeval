from typing import List, Literal
from deepteam.frameworks import RedTeamingFramework
from deepteam.frameworks.eu_ai_act.risk_categories import (
    EU_AI_ACT_CATEGORIES,
)

"""
EU Artificial Intelligence Act (Regulation (EU) 2024/1689)
==========================================================

The EU AI Act is the world's first comprehensive legal framework for artificial intelligence.
It adopts a risk-based approach, classifying AI systems by the level of risk they pose to
health, safety, and fundamental rights, and imposing proportionate obligations on providers
and deployers.

DeepTeam's EU AI Act module operationalises two of the central risk tiers of the Act so you
can red-team AI systems against the obligations that matter most in practice:

- Article 5 — Prohibited AI practices (unacceptable risk):
    1. Art. 5(1)(a) Subliminal Manipulation
    2. Art. 5(1)(b) Exploitation of Vulnerabilities
    3. Art. 5(1)(c) Social Scoring
    4. Art. 5(1)(g) Biometric Categorisation
    5. Art. 5(1)(h) Real-time Remote Biometric Identification
    6. Art. 5      Post Remote Biometric Identification

- Annex III — High-risk AI systems (Art. 6(2)):
    1. §1 Biometric Identification
    2. §2 Critical Infrastructure
    3. §3 Education and Vocational Training
    4. §4 Employment and Workers Management
    5. §5 Essential Private and Public Services
    6. §6 Law Enforcement
    7. §7 Migration, Asylum and Border Control
    8. §8 Administration of Justice and Democratic Processes

Each category includes:
- Attacks: Adversarial techniques chosen to stress the specific risk addressed by that article
- Vulnerabilities: DeepTeam vulnerability classes whose behaviours correspond to the article

Reference: https://artificialintelligenceact.eu/
"""


class EUAIAct(RedTeamingFramework):
    name = "EU Artificial Intelligence Act"
    description = "A risk-based AI regulation operationalised in DeepTeam across Article 5 prohibited practices (subliminal manipulation, exploitation of vulnerabilities, social scoring, biometric categorisation, real-time and post remote biometric identification) and Annex III high-risk AI systems (biometric identification, critical infrastructure, education, employment, essential services, law enforcement, migration/border control, and administration of justice and democracy)."
    ALLOWED_TYPES = [
        "subliminal_manipulation",
        "exploitation_of_vulnerabilities",
        "social_scoring",
        "biometric_categorisation",
        "remote_biometric_id_live",
        "remote_biometric_id_post",
        "biometric_id",
        "critical_infrastructure",
        "education",
        "employment",
        "essential_services",
        "law_enforcement",
        "migration_border",
        "justice_democracy",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "subliminal_manipulation",
                "exploitation_of_vulnerabilities",
                "social_scoring",
                "biometric_categorisation",
                "remote_biometric_id_live",
                "remote_biometric_id_post",
                "biometric_id",
                "critical_infrastructure",
                "education",
                "employment",
                "essential_services",
                "law_enforcement",
                "migration_border",
                "justice_democracy",
            ]
        ] = [
            "subliminal_manipulation",
            "exploitation_of_vulnerabilities",
            "social_scoring",
            "biometric_categorisation",
            "remote_biometric_id_live",
            "remote_biometric_id_post",
            "biometric_id",
            "critical_infrastructure",
            "education",
            "employment",
            "essential_services",
            "law_enforcement",
            "migration_border",
            "justice_democracy",
        ],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []
        for category in categories:
            for risk_category in EU_AI_ACT_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
