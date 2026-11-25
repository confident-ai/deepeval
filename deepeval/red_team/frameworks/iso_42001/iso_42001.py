"""
ISO/IEC 42001:2023 — AI Management Systems framework mapper.

See ``risk_categories`` for the in-scope vs. out-of-scope rationale and
the per-control vulnerability/attack mapping.
"""

from typing import List, Literal

from deepeval.red_team.attacks import BaseAttack
from deepeval.red_team.frameworks.frameworks import RedTeamingFramework
from deepeval.red_team.frameworks.iso_42001.risk_categories import (
    ISO_42001_CATEGORIES,
)
from deepeval.red_team.vulnerabilities import BaseVulnerability


class ISO42001(RedTeamingFramework):
    name = "ISO/IEC 42001:2023 — AI Management Systems"
    description = (
        "Red-team evidence aligned to seven outcome-focused ISO/IEC "
        "42001:2023 Annex A controls: fairness (A.6.2.4), robustness "
        "(A.6.2.5), security (A.6.2.6), privacy (A.6.2.7), transparency "
        "(A.6.2.8), safety and ethical use (A.9.2), and accountability "
        "with human oversight (A.9.3)."
    )

    ALLOWED_TYPES = [
        "A.6.2.4",
        "A.6.2.5",
        "A.6.2.6",
        "A.6.2.7",
        "A.6.2.8",
        "A.9.2",
        "A.9.3",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "A.6.2.4",
                "A.6.2.5",
                "A.6.2.6",
                "A.6.2.7",
                "A.6.2.8",
                "A.9.2",
                "A.9.3",
            ]
        ] = None,
    ):
        if categories is None:
            categories = list(self.ALLOWED_TYPES)

        invalid = [c for c in categories if c not in self.ALLOWED_TYPES]
        if invalid:
            raise ValueError(
                f"Unknown ISO/IEC 42001 control id(s): {invalid}. "
                f"Allowed: {self.ALLOWED_TYPES}"
            )

        # Map the user-facing control id to the internal RiskCategory key.
        control_to_key = {
            "A.6.2.4": "a_6_2_4_fairness",
            "A.6.2.5": "a_6_2_5_reliability_robustness",
            "A.6.2.6": "a_6_2_6_security",
            "A.6.2.7": "a_6_2_7_privacy",
            "A.6.2.8": "a_6_2_8_transparency",
            "A.9.2": "a_9_2_responsible_use",
            "A.9.3": "a_9_3_human_oversight",
        }
        wanted_keys = {control_to_key[c] for c in categories}

        self.categories = categories
        self.risk_categories = [
            rc for rc in ISO_42001_CATEGORIES if rc.name in wanted_keys
        ]

        vulnerabilities: List[BaseVulnerability] = []
        attacks: List[BaseAttack] = []
        for rc in self.risk_categories:
            vulnerabilities.extend(rc.vulnerabilities)
            attacks.extend(rc.attacks)

        self.vulnerabilities = vulnerabilities
        self.attacks = attacks

    def get_name(self) -> str:
        return self.name
