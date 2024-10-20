from enum import Enum
from typing import Dict


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


class UnalignedVulnerability(Enum):
    VIOLENT_CRIME = "Violent Crimes"
    NON_VIOLENT_CRIME = "Non Violent Crimes"
    SEX_CRIME = "Sex Crimes"
    CHILD_EXPLOITATION = "Child Exploitation"
    INDISCRIMINATE_WEAPONS = "Indiscriminate Weapons"
    HATE = "Hate"
    SELF_HARM = "Self Harm"
    SEXUAL_CONTENT = "Sexual Content"
    CYBERCRIME = "Cybercrime"
    CHEMICAL_BIOLOGICAL_WEAPONS = "Chemical & Biological Weapons"
    ILLEGAL_DRUGS = "Illegal Drugs"
    COPYRIGHT_VIOLATIONS = "Copyright Violations"
    HARASSMENT_BULLYING = "Harassment & Bullying"
    ILLEGAL_ACTIVITIES = "Illegal Activities"
    GRAPHIC_CONTENT = "Graphic Content"
    UNSAFE_PRACTICES = "Unsafe Practices"
    RADICALIZATION = "Radicalization"
    PROFANITY = "Profanity"
    INSULTS = "Insults"


class RemoteVulnerability(Enum):
    SSRF = "Server Side Request Forgery"
    BOLA = "Broken Object Level Authorization"
    BFLA = "Broken Function Level Authorization"
    COMPETITORS = "Competitors"
    HIJACKING = "Hijacking"
    RELIGION = "Religion"


class Vulnerability(Enum):
    PRIVACY = "Privacy"
    INTELLECTUAL_PROPERTY = "Intellectual Property"
    MISINFORMATION_DISINFORMATION = "Misinformation & Disinformation"
    SPECIALIZED_FINANCIAL_ADVICE = "Specialized Financial Advice"
    OFFENSIVE = "Offensive"
    BIAS = "BIAS"
    PII_API_DB = "API and Database Access"
    PII_DIRECT = "Direct PII Disclosure"
    PII_SESSION = "Session PII Leak"
    PII_SOCIAL = "Social Engineering PII Disclosure"
    DATA_LEAKAGE = "Data Leakage"
    CONTRACTS = "Contracts"
    EXCESSIVE_AGENCY = "Excessive Agency"
    HALLUCINATION = "Hallucination"
    IMITATION = "Imitation"
    POLITICS = "Political Statements"
    OVERRELIANCE = "Overreliance"
    DEBUG_ACCESS = "Debug Access"
    RBAC = "Role-Based Access Control"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"
    PROMPT_EXTRACTION = "Prompt Extraction"

    RELIGION = RemoteVulnerability.RELIGION.value
    HIJACKING = RemoteVulnerability.HIJACKING.value
    COMPETITORS = RemoteVulnerability.COMPETITORS.value
    SSRF = RemoteVulnerability.SSRF.value
    BFLA = RemoteVulnerability.BFLA.value
    BOLA = RemoteVulnerability.BOLA.value

    VIOLENT_CRIME = UnalignedVulnerability.VIOLENT_CRIME.value
    NON_VIOLENT_CRIME = UnalignedVulnerability.NON_VIOLENT_CRIME.value
    SEX_CRIME = UnalignedVulnerability.SEX_CRIME.value
    CHILD_EXPLOITATION = UnalignedVulnerability.CHILD_EXPLOITATION.value
    INDISCRIMINATE_WEAPONS = UnalignedVulnerability.INDISCRIMINATE_WEAPONS.value
    HATE = UnalignedVulnerability.HATE.value
    SELF_HARM = UnalignedVulnerability.SELF_HARM.value
    SEXUAL_CONTENT = UnalignedVulnerability.SEXUAL_CONTENT.value
    CYBERCRIME = UnalignedVulnerability.CYBERCRIME.value
    ILLEGAL_DRUGS = UnalignedVulnerability.ILLEGAL_DRUGS.value
    COPYRIGHT_VIOLATIONS = UnalignedVulnerability.COPYRIGHT_VIOLATIONS.value
    HARASSMENT_BULLYING = UnalignedVulnerability.HARASSMENT_BULLYING.value
    ILLEGAL_ACTIVITIES = UnalignedVulnerability.ILLEGAL_ACTIVITIES.value
    GRAPHIC_CONTENT = UnalignedVulnerability.GRAPHIC_CONTENT.value
    UNSAFE_PRACTICES = UnalignedVulnerability.UNSAFE_PRACTICES.value
    RADICALIZATION = UnalignedVulnerability.RADICALIZATION.value
    PROFANITY = UnalignedVulnerability.PROFANITY.value
    INSULTS = UnalignedVulnerability.INSULTS.value
    CHEMICAL_BIOLOGICAL_WEAPONS = (
        UnalignedVulnerability.CHEMICAL_BIOLOGICAL_WEAPONS.value
    )


class LLMRiskCategories(Enum):
    RESPONSIBLE_AI = "Responsible AI"
    ILLEGAL = "Illegal"
    BRAND_IMAGE = "Brand Image"
    DATA_PRIVACY = "Data Privacy"
    UNAUTHORIZED_ACCESS = "Unauthorized Access"


llm_risk_categories_map: Dict[Vulnerability, LLMRiskCategories] = {
    # Responsible AI
    Vulnerability.BIAS: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.POLITICS: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.RELIGION: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.HATE: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.RADICALIZATION: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.OFFENSIVE: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.PROFANITY: LLMRiskCategories.RESPONSIBLE_AI,
    Vulnerability.INSULTS: LLMRiskCategories.RESPONSIBLE_AI,
    # Illegal
    Vulnerability.VIOLENT_CRIME: LLMRiskCategories.ILLEGAL,
    Vulnerability.NON_VIOLENT_CRIME: LLMRiskCategories.ILLEGAL,
    Vulnerability.SEX_CRIME: LLMRiskCategories.ILLEGAL,
    Vulnerability.CYBERCRIME: LLMRiskCategories.ILLEGAL,
    Vulnerability.CHILD_EXPLOITATION: LLMRiskCategories.ILLEGAL,
    Vulnerability.ILLEGAL_DRUGS: LLMRiskCategories.ILLEGAL,
    Vulnerability.ILLEGAL_ACTIVITIES: LLMRiskCategories.ILLEGAL,
    Vulnerability.UNSAFE_PRACTICES: LLMRiskCategories.ILLEGAL,
    Vulnerability.SELF_HARM: LLMRiskCategories.ILLEGAL,
    Vulnerability.HARASSMENT_BULLYING: LLMRiskCategories.ILLEGAL,
    Vulnerability.SEXUAL_CONTENT: LLMRiskCategories.ILLEGAL,
    Vulnerability.GRAPHIC_CONTENT: LLMRiskCategories.ILLEGAL,
    Vulnerability.CHEMICAL_BIOLOGICAL_WEAPONS: LLMRiskCategories.ILLEGAL,
    Vulnerability.INDISCRIMINATE_WEAPONS: LLMRiskCategories.ILLEGAL,
    Vulnerability.COPYRIGHT_VIOLATIONS: LLMRiskCategories.ILLEGAL,
    Vulnerability.INTELLECTUAL_PROPERTY: LLMRiskCategories.ILLEGAL,
    # Brand Image
    Vulnerability.COMPETITORS: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.IMITATION: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.MISINFORMATION_DISINFORMATION: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.HALLUCINATION: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.OVERRELIANCE: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.SPECIALIZED_FINANCIAL_ADVICE: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.CONTRACTS: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.EXCESSIVE_AGENCY: LLMRiskCategories.BRAND_IMAGE,
    Vulnerability.HIJACKING: LLMRiskCategories.BRAND_IMAGE,
    # Data Privacy
    Vulnerability.PII_API_DB: LLMRiskCategories.DATA_PRIVACY,
    Vulnerability.PII_DIRECT: LLMRiskCategories.DATA_PRIVACY,
    Vulnerability.PII_SESSION: LLMRiskCategories.DATA_PRIVACY,
    Vulnerability.PII_SOCIAL: LLMRiskCategories.DATA_PRIVACY,
    Vulnerability.DATA_LEAKAGE: LLMRiskCategories.DATA_PRIVACY,
    Vulnerability.PRIVACY: LLMRiskCategories.DATA_PRIVACY,
    # Unauthorized Access
    Vulnerability.DEBUG_ACCESS: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.RBAC: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.SHELL_INJECTION: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.SQL_INJECTION: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.SSRF: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.BFLA: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.BOLA: LLMRiskCategories.UNAUTHORIZED_ACCESS,
    Vulnerability.PROMPT_EXTRACTION: LLMRiskCategories.UNAUTHORIZED_ACCESS,
}

remote_vulnerability_to_api_code_map = {
    Vulnerability.SSRF: "ssrf",
    Vulnerability.BOLA: "bola",
    Vulnerability.BFLA: "bfla",
    Vulnerability.COMPETITORS: "competitors",
    Vulnerability.HIJACKING: "hijacking",
    Vulnerability.RELIGION: "religion",
}

unaligned_vulnerability_to_api_code_map = {
    Vulnerability.VIOLENT_CRIME: "Violent Crimes",
    Vulnerability.NON_VIOLENT_CRIME: "Non-Violent Crimes",
    Vulnerability.SEX_CRIME: "Sex Crimes",
    Vulnerability.CHILD_EXPLOITATION: "Child Exploitation",
    Vulnerability.INDISCRIMINATE_WEAPONS: "Indiscriminate Weapons",
    Vulnerability.HATE: "Hate",
    Vulnerability.SELF_HARM: "Self-Harm",
    Vulnerability.SEXUAL_CONTENT: "Sexual Content",
    Vulnerability.CYBERCRIME: "Cybercrime & Unauthorized Intrusion - Hacking and Malware",
    Vulnerability.CHEMICAL_BIOLOGICAL_WEAPONS: "Chemical & Biological Weapons",
    Vulnerability.ILLEGAL_DRUGS: "Illegal Drugs",
    Vulnerability.COPYRIGHT_VIOLATIONS: "Copyright Violations - Copyrighted text",
    Vulnerability.HARASSMENT_BULLYING: "Harassment & Bullying",
    Vulnerability.ILLEGAL_ACTIVITIES: "Illegal Activities - Fraud & scams",
    Vulnerability.GRAPHIC_CONTENT: "Graphic & age-restricted content",
    Vulnerability.UNSAFE_PRACTICES: "Promotion of unsafe practices",
    Vulnerability.RADICALIZATION: "Radicalization",
    Vulnerability.PROFANITY: "Requests containing profanity",
    Vulnerability.INSULTS: "Insults and personal attacks",
}
