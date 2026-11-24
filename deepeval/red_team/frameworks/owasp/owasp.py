from typing import List, Literal
from deepteam.frameworks import RedTeamingFramework
from deepteam.vulnerabilities import (
    BaseVulnerability,
)
from deepteam.attacks import BaseAttack
from deepteam.frameworks.owasp.risk_categories import OWASP_CATEGORIES

"""
OWASP Top 10 for Large Language Models (LLMs) 2025
===================================================

The OWASP Top 10 for Large Language Models is a comprehensive list of the most critical 
security risks associated with LLM applications. This resource is designed to help 
developers, security professionals, and organizations identify, understand, and mitigate 
vulnerabilities in LLM systems, ensuring safer and more robust deployments in real-world 
applications.

The 2025 edition reflects significant evolution in the LLM threat landscape, with new 
risks emerging from RAG systems, autonomous AI agents, and sophisticated attack methods.

The 10 critical risks are:
1. LLM01 - Prompt Injection: Manipulating LLM inputs to override instructions
2. LLM02 - Sensitive Information Disclosure: Unintended exposure of private data
3. LLM03 - Supply Chain: Compromised third-party components, models, or plugins
4. LLM04 - Data and Model Poisoning: Manipulation of training or fine-tuning data
5. LLM05 - Improper Output Handling: Inadequate validation of LLM outputs
6. LLM06 - Excessive Agency: Too much autonomy or permissions granted to LLMs
7. LLM07 - System Prompt Leakage: Exposure of internal prompts and credentials (NEW 2025)
8. LLM08 - Vector and Embedding Weaknesses: RAG and vector database vulnerabilities (NEW 2025)
9. LLM09 - Misinformation: False information and hallucinations
10. LLM10 - Unbounded Consumption: Uncontrolled resource usage

Each category includes:
- Attacks: Realistic red-teaming techniques for prompting, probing, or jailbreaking
- Vulnerabilities: Typed definitions capturing observable weaknesses and security flaws

Reference: https://genai.owasp.org/llm-top-10/
"""


class OWASPTop10(RedTeamingFramework):
    name = "OWASP Top 10 for LLMs 2025"
    description = "A comprehensive list of the most critical security risks associated with LLM applications. The 2025 edition includes 10 critical risks covering prompt injection, sensitive information disclosure, supply chain vulnerabilities, data poisoning, output handling, excessive agency, system prompt leakage, vector/embedding weaknesses, misinformation, and unbounded consumption. Each risk is tested using realistic attack techniques and vulnerability assessments."
    ALLOWED_TYPES = [
        "LLM_01",
        "LLM_02",
        "LLM_03",
        "LLM_04",
        "LLM_05",
        "LLM_06",
        "LLM_07",
        "LLM_08",
        "LLM_09",
        "LLM_10",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "LLM_01",
                "LLM_02",
                "LLM_03",
                "LLM_04",
                "LLM_05",
                "LLM_06",
                "LLM_07",
                "LLM_08",
                "LLM_09",
                "LLM_10",
            ]
        ] = [
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        ],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []
        for category in categories:
            for risk_category in OWASP_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
