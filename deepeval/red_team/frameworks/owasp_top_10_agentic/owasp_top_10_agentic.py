from typing import List, Literal
from deepteam.frameworks import RedTeamingFramework
from deepteam.frameworks.owasp_top_10_agentic.risk_categories import (
    OWASP_ASI_CATEGORIES,
)

"""
OWASP Top 10 for Agentic Applications (ASI) 2026
===============================================

The OWASP Top 10 for Agentic Applications identifies the most critical security risks 
introduced by autonomous and semi-autonomous AI agents. Unlike traditional LLM 
applications, agentic systems combine reasoning, memory, tools, and multi-step execution,
introducing new classes of vulnerabilities that extend beyond prompt-level attacks.

The 2026 edition focuses on failures arising from goal misalignment, tool misuse,
delegated trust, inter-agent communication, persistent memory, and emergent autonomous
behavior.

The 10 critical risks are:
1. ASI01 - Agent Goal Hijack: Manipulation of agent objectives or plans
2. ASI02 - Tool Misuse & Exploitation: Unsafe or excessive use of tools
3. ASI03 - Agent Identity & Privilege Abuse: Misuse of delegated authority or trust
4. ASI04 - Agentic Supply Chain Compromise: Malicious tools, agents, or metadata
5. ASI05 - Unexpected Code Execution: Unsafe execution of agent-generated code
6. ASI06 - Memory & Context Poisoning: Corruption of agent memory or contextual state
7. ASI07 - Insecure Inter-Agent Communication: Manipulated agent-to-agent messages
8. ASI08 - Cascading Agent Failures: Failure propagation across agent systems
9. ASI09 - Human-Agent Trust Exploitation: Abuse of human over-reliance on agents
10. ASI10 - Rogue Agents: Agents acting beyond intended objectives

Each category includes:
- Attacks: Prompting and interaction patterns that induce unsafe agent behavior
- Vulnerabilities: Typed behavioral weaknesses observable at the model or agent layer

This framework evaluates agentic risks at the reasoning and behavioral level and does not
claim full coverage of runtime enforcement, infrastructure security, or cryptographic
protocols.

Reference: https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
"""


class OWASP_ASI_2026(RedTeamingFramework):
    name = "OWASP Top 10 for Agentic Applications 2026"
    description = "A comprehensive list of the most critical security risks associated with agentic AI applications. The 2026 edition focuses on failures introduced by autonomy, tool usage, delegated trust, memory persistence, inter-agent communication, and emergent behavior. Each risk category is evaluated using realistic attack techniques and typed vulnerability assessments aligned with agent reasoning and behavior."

    ALLOWED_TYPES = [
        "ASI_01",
        "ASI_02",
        "ASI_03",
        "ASI_04",
        "ASI_05",
        "ASI_06",
        "ASI_07",
        "ASI_08",
        "ASI_09",
        "ASI_10",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "ASI_01",
                "ASI_02",
                "ASI_03",
                "ASI_04",
                "ASI_05",
                "ASI_06",
                "ASI_07",
                "ASI_08",
                "ASI_09",
                "ASI_10",
            ]
        ] = [
            "ASI_01",
            "ASI_02",
            "ASI_03",
            "ASI_04",
            "ASI_05",
            "ASI_06",
            "ASI_07",
            "ASI_08",
            "ASI_09",
            "ASI_10",
        ],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []

        for category in categories:
            for risk_category in OWASP_ASI_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
