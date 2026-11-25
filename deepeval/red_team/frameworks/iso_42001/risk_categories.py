"""
ISO/IEC 42001:2023 Annex A risk categories — focused profile.

Scoped to seven outcome themes that can be exercised via red-team:

  * Fairness & Bias Prevention            -> A.6.2.4
  * Robustness & Resilience               -> A.6.2.5
  * Security & Vulnerability Management   -> A.6.2.6
  * Privacy & Data Protection             -> A.6.2.7
  * Transparency & Trustworthiness        -> A.6.2.8
  * Safety & Ethical Use                  -> A.9.2
  * Accountability & Human Oversight      -> A.9.3

All other Annex A controls (data lineage A.7.2/A.7.4, external reporting
A.8.3, intended use A.9.4, third-party AI A.10.2) and pure-governance
controls (A.4.2, A.6.2.3, A.7.3, A.8.2, A.10.3/A.10.4) are intentionally
omitted from this profile. They are out of scope for behavioral
red-team evidence and require governance artifacts instead.

Reference: ISO/IEC 42001:2023, Annex A.
"""

from deepeval.red_team.attacks.multi_turn import (
    BadLikertJudge,
    CrescendoJailbreaking,
    LinearJailbreaking,
    SequentialJailbreak,
    TreeJailbreaking,
)
from deepeval.red_team.attacks.single_turn import (
    AuthorityEscalation,
    Base64,
    ContextPoisoning,
    EmotionalManipulation,
    GoalRedirection,
    Multilingual,
    PromptInjection,
    Roleplay,
    SystemOverride,
)
from deepeval.red_team.frameworks.risk_category import RiskCategory
from deepeval.red_team.vulnerabilities import (
    BFLA,
    BOLA,
    PIILeakage,
    RBAC,
    SQLInjection,
    SSRF,
    Bias,
    DebugAccess,
    ExcessiveAgency,
    Hallucination,
    IllegalActivity,
    Misinformation,
    PromptLeakage,
    Robustness,
    ShellInjection,
    Toxicity,
)


# -------------------------------------------------------- Fairness & Bias Prevention
a_6_2_4_fairness = RiskCategory(
    name="a_6_2_4_fairness",
    _display_name="A.6.2.4 — Fairness & Bias Prevention",
    description=(
        "The AI system avoids unfair bias against protected groups. "
        "Probes test for disparate treatment across race, gender, age, "
        "religion, and political orientation under direct, multilingual, "
        "and emotionally-charged framings."
    ),
    vulnerabilities=[
        Bias(types=["race", "gender", "religion", "politics"]),
        Toxicity(types=["insults", "profanity", "threats", "mockery"]),
    ],
    attacks=[
        Roleplay(weight=2),
        Multilingual(weight=1),
        EmotionalManipulation(weight=1),
        LinearJailbreaking(weight=1),
    ],
)


# ---------------------------------------------------------- Robustness & Resilience
a_6_2_5_reliability_robustness = RiskCategory(
    name="a_6_2_5_reliability_robustness",
    _display_name="A.6.2.5 — Robustness & Resilience",
    description=(
        "The AI system performs consistently under expected and "
        "adversarial conditions. Probes stress output stability through "
        "context poisoning, goal redirection, and progressive multi-turn "
        "attacks."
    ),
    vulnerabilities=[
        Robustness(types=["input_overreliance", "hijacking"]),
        Hallucination(),
        Misinformation(
            types=[
                "factual_errors",
                "unsupported_claims",
                "expertize_misrepresentation",
            ]
        ),
    ],
    attacks=[
        ContextPoisoning(weight=2),
        GoalRedirection(weight=1),
        SequentialJailbreak(weight=1),
        CrescendoJailbreaking(weight=2),
        TreeJailbreaking(weight=1),
    ],
)


# ------------------------------------------- Security & Vulnerability Management
a_6_2_6_security = RiskCategory(
    name="a_6_2_6_security",
    _display_name="A.6.2.6 — Security & Vulnerability Management",
    description=(
        "The AI system is protected against adversarial threats and "
        "unauthorized actions. Covers prompt leakage, injection-style "
        "attacks against tool-using agents, and access-control bypass."
    ),
    vulnerabilities=[
        PromptLeakage(types=["secrets_and_credentials", "instructions"]),
        ShellInjection(),
        SQLInjection(),
        SSRF(),
        DebugAccess(),
        RBAC(),
        BFLA(),
        BOLA(),
    ],
    attacks=[
        PromptInjection(weight=2),
        Base64(weight=1),
        AuthorityEscalation(weight=1),
        SystemOverride(weight=1),
        LinearJailbreaking(weight=1),
        TreeJailbreaking(weight=1),
        BadLikertJudge(weight=1),
    ],
)


# ----------------------------------------------------- Privacy & Data Protection
a_6_2_7_privacy = RiskCategory(
    name="a_6_2_7_privacy",
    _display_name="A.6.2.7 — Privacy & Data Protection",
    description=(
        "Personal data is processed lawfully, minimally, and is not "
        "leaked. Probes target direct, indirect, and cross-context PII "
        "exposure as well as system-prompt extraction."
    ),
    vulnerabilities=[
        PIILeakage(
            types=[
                "direct_disclosure",
                "session_leak",
                "social_manipulation",
            ]
        ),
        PromptLeakage(types=["secrets_and_credentials"]),
    ],
    attacks=[
        Roleplay(weight=2),
        EmotionalManipulation(weight=1),
        Multilingual(weight=1),
        CrescendoJailbreaking(weight=2),
    ],
)


# ----------------------------------------------- Transparency & Trustworthiness
a_6_2_8_transparency = RiskCategory(
    name="a_6_2_8_transparency",
    _display_name="A.6.2.8 — Transparency & Trustworthiness",
    description=(
        "System behavior is explainable to relevant stakeholders. "
        "Behavioral evidence that the system does not fabricate "
        "explanations or invent reasoning."
    ),
    vulnerabilities=[
        Hallucination(),
        Misinformation(types=["unsupported_claims"]),
    ],
    attacks=[
        PromptInjection(weight=1),
        ContextPoisoning(weight=1),
    ],
)


# ----------------------------------------------------------- Safety & Ethical Use
a_9_2_responsible_use = RiskCategory(
    name="a_9_2_responsible_use",
    _display_name="A.9.2 — Safety & Ethical Use",
    description=(
        "Operational processes prevent misuse and ensure responsible, "
        "ethical use. Probes test compliance with usage boundaries "
        "under social-engineering and escalating multi-turn pressure."
    ),
    vulnerabilities=[
        ExcessiveAgency(types=["functionality", "permissions", "autonomy"]),
        IllegalActivity(
            types=[
                "weapons",
                "cybercrime",
                "non_violent_crimes",
            ]
        ),
    ],
    attacks=[
        AuthorityEscalation(weight=2),
        GoalRedirection(weight=1),
        CrescendoJailbreaking(weight=2),
        SequentialJailbreak(weight=1),
    ],
)


# -------------------------------------------- Accountability & Human Oversight
a_9_3_human_oversight = RiskCategory(
    name="a_9_3_human_oversight",
    _display_name="A.9.3 — Accountability & Human Oversight",
    description=(
        "Meaningful human oversight and accountability over AI "
        "decisions. Probes attempt to remove, skip, or silently bypass "
        "human-in-the-loop checkpoints."
    ),
    vulnerabilities=[
        ExcessiveAgency(types=["autonomy", "permissions"]),
    ],
    attacks=[
        AuthorityEscalation(weight=2),
        GoalRedirection(weight=2),
        SystemOverride(weight=1),
        CrescendoJailbreaking(weight=1),
        TreeJailbreaking(weight=1),
    ],
)


ISO_42001_CATEGORIES = [
    a_6_2_4_fairness,
    a_6_2_5_reliability_robustness,
    a_6_2_6_security,
    a_6_2_7_privacy,
    a_6_2_8_transparency,
    a_9_2_responsible_use,
    a_9_3_human_oversight,
]
