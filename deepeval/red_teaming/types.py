from typing import Dict, Optional, List
from pydantic import BaseModel
from enum import Enum

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

    SSRF = "Server Side Request Forgery"
    BOLA = "Broken Object Level Authorization"
    BFLA = "Broken Function Level Authorization"
    COMPETITORS = "Competitors"
    HIJACKING = "Hijacking"
    RELIGION = "Religion"

    VIOLENT_CRIME = "Violent Crimes"
    NON_VIOLENT_CRIME = "Non Violent Crimes"
    SEX_CRIME = "Sex Crimes"
    CHILD_EXPLOITATION = "Child Exploitation"
    SELF_HARM = "Self Harm"
    SEXUAL_CONTENT = "Sexual Content"
    CYBERCRIME = "Cybercrime"
    WEAPONS = "Weapons"
    ILLEGAL_DRUGS = "Illegal Drugs"
    COPYRIGHT_VIOLATIONS = "Copyright Violations"
    HARASSMENT_BULLYING = "Harassment & Bullying"
    ILLEGAL_ACTIVITIES = "Illegal Activities"
    GRAPHIC_CONTENT = "Graphic Content"
    UNSAFE_PRACTICES = "Unsafe Practices"
    PROFANITY = "Profanity"
    INSULTS = "Insults"

remote_vulnerabilties = [
    Vulnerability.SSRF,
    Vulnerability.BOLA,
    Vulnerability.BFLA,
    Vulnerability.COMPETITORS,
    Vulnerability.HIJACKING,
    Vulnerability.RELIGION,
    Vulnerability.VIOLENT_CRIME,
    Vulnerability.NON_VIOLENT_CRIME,
    Vulnerability.SEX_CRIME,
    Vulnerability.CHILD_EXPLOITATION,
    Vulnerability.SELF_HARM,
    Vulnerability.SEXUAL_CONTENT,
    Vulnerability.CYBERCRIME,
    Vulnerability.ILLEGAL_DRUGS,
    Vulnerability.COPYRIGHT_VIOLATIONS,
    Vulnerability.HARASSMENT_BULLYING,
    Vulnerability.ILLEGAL_ACTIVITIES,
    Vulnerability.GRAPHIC_CONTENT,
    Vulnerability.PROFANITY,
    Vulnerability.INSULTS,
]

class Attack(BaseModel):
    vulnerability: Vulnerability
    # When there is an error, base input can fail to generate
    # and subsequently enhancements are redundant
    input: Optional[str] = None
    attack_enhancement: Optional[str] = None
    error: Optional[str] = None

class ApiGenerateBaselineAttack(BaseModel):
    purpose: str
    vulnerability: str
    num_attacks: int

class GenerateBaselineAttackResponseData(BaseModel):
    baseline_attacks: List[Attack]

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