from typing import List, Literal
from deepteam.frameworks import RedTeamingFramework
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.nist.risk_categories import NIST_CATEGORIES

"""
NIST AI Risk Management Framework (AI RMF) 1.0
===============================================

The NIST AI Risk Management Framework (AI RMF) is a structured methodology from the U.S. National 
Institute of Standards and Technology that guides organizations in identifying, evaluating, and 
mitigating risks in artificial intelligence systems. It promotes trustworthy AI by focusing on 
governance, measurement, and continuous risk tracking across the AI lifecycle.

DeepTeam's implementation focuses on the Measure function (M.1-M.4) — the part of the framework 
responsible for testing, evaluation, and assurance of AI behavior and risk controls.

The framework is divided into four measurement categories:
- Measure 1 (M.1): Risk measurement and metrics - Define and apply appropriate testing and metrics for AI risk evaluation
- Measure 2 (M.2): Trustworthiness and safety evaluation - Evaluate AI systems for trustworthiness, safety, security, fairness, and misuse potential
- Measure 3 (M.3): Risk tracking and monitoring - Establish mechanisms for identifying, tracking, and managing emerging risks
- Measure 4 (M.4): Impact and transparency assessment - Measure and correlate AI risk impacts with business and performance outcomes

Each category includes:
- Attacks: Techniques for testing AI system resilience and detecting vulnerabilities
- Vulnerabilities: Weaknesses that can be exploited in AI systems

Reference: https://www.nist.gov/itl/ai-risk-management-framework
"""


class NIST(RedTeamingFramework):
    name = "NIST AI Risk Management Framework (AI RMF)"
    description = "A structured methodology from NIST for identifying, evaluating, and mitigating risks in AI systems. Confident AI's implementation focuses on the Measure function - testing, evaluation, and assurance of AI behavior and risk controls across four measurement categories (M.1-M.4) covering risk metrics, trustworthiness evaluation, risk tracking, and impact assessment."
    ALLOWED_TYPES = ["measure_1", "measure_2", "measure_3", "measure_4"]

    def __init__(
        self,
        categories: List[
            Literal["measure_1", "measure_2", "measure_3", "measure_4"]
        ] = ["measure_1", "measure_2", "measure_3", "measure_4"],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []
        for category in categories:
            for risk_category in NIST_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
