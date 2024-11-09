from enum import Enum


class Guard(Enum):
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


# Lists of guards that require purpose, entities, or both
purpose_dependent_guards = [
    Guard.BFLA,
    Guard.BIAS,
    Guard.HALLUCINATION,
    Guard.HIJACKING,
    Guard.OVERRELIANCE,
    Guard.PROMPT_EXTRACTION,
    Guard.RBAC,
    Guard.SSRF,
    Guard.COMPETITORS,
    Guard.RELIGION,
]

entities_dependent_guards = [Guard.BOLA, Guard.IMITATION]

purpose_entities_dependent_guards = [
    Guard.PII_API_DB,
    Guard.PII_DIRECT,
    Guard.PII_SESSION,
    Guard.PII_SOCIAL,
    Guard.COMPETITORS,
    Guard.RELIGION,
]
