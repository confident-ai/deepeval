from enum import Enum


class UseCase(Enum):
    QA = "QA"
    TEXT2SQL = "Text to SQL"


class Evolution(Enum):
    REASONING = "Reasoning"
    MULTICONTEXT = "Multi-context"
    CONCRETIZING = "Concretizing"
    CONSTRAINED = "Constrained"
    COMPARATIVE = "Comparative"
    HYPOTHETICAL = "Hypothetical"
    IN_BREADTH = "In-Breadth"


class PromptEvolution(Enum):
    REASONING = "Reasoning"
    CONCRETIZING = "Concretizing"
    CONSTRAINED = "Constrained"
    COMPARATIVE = "Comparative"
    HYPOTHETICAL = "Hypothetical"
    IN_BREADTH = "In-Breadth"


class RTAdversarialAttack(Enum):
    GRAY_BOX_ATTACK = "Gray Box Attack"
    PROMPT_INJECTION = "Prompt Injection"
    PROMPT_PROBING = "Prompt Probing"
    JAILBREAKING = "Jailbreaking"
    JAILBREAK_LINEAR = "Linear Jailbreak"
    JAILBREAK_TREE = "Tree Jailbreak"
    ROT13 = "ROT13 Encoding"
    BASE64 = "Base64 Encoding"
    LEETSPEAK = "Leetspeak Encoding"


class RTUnalignedVulnerabilities(Enum):
    HARMFUL_VIOLENT_CRIME = "Violent Crimes"
    HARMFUL_NON_VIOLENT_CRIME = "Non-Violent Crimes"
    HARMFUL_SEX_CRIME = "Sex Crimes"
    HARMFUL_CHILD_EXPLOITATION = "Child Exploitation"
    HARMFUL_INDISCRIMINATE_WEAPONS = "Indiscriminate Weapons"
    HARMFUL_HATE = "Hate"
    HARMFUL_SELF_HARM = "Self-Harm"
    HARMFUL_SEXUAL_CONTENT = "Sexual Content"
    HARMFUL_CYBERCRIME = (
        "Cybercrime & Unauthorized Intrusion - Hacking and Malware"
    )
    HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS = "Chemical & Biological Weapons"
    HARMFUL_ILLEGAL_DRUGS = "Illegal Drugs"
    HARMFUL_COPYRIGHT_VIOLATIONS = "Copyright Violations - Copyrighted text"
    HARMFUL_HARASSMENT_BULLYING = "Harassment & Bullying"
    HARMFUL_ILLEGAL_ACTIVITIES = "Illegal Activities - Fraud & scams"
    HARMFUL_GRAPHIC_CONTENT = "Graphic & age-restricted content"
    HARMFUL_UNSAFE_PRACTICES = "Promotion of unsafe practices"
    HARMFUL_RADICALIZATION = "Radicalization"
    HARMFUL_PROFANITY = "Requests containing profanity"
    HARMFUL_INSULTS = "Insults and personal attacks"


class RTVulnerability(Enum):
    OFFENSIVE = "Generate harmful content"
    BIAS = "Promote stereotypes and discrimination"
    DATA_LEAKAGE = "Leak confidential data and information"
    UNFORMATTED = "Output undesirable formats"

    # Harm Vulnerabilties
    HARMFUL_PRIVACY = "Privacy violations"
    HARMFUL_INTELLECTUAL_PROPERTY = "Intellectual Property violation"
    HARMFUL_MISINFORMATION_DISINFORMATION = (
        "Misinformation & Disinformation - Harmful lies and propaganda"
    )
    HARMFUL_SPECIALIZED_ADVICE_FINANCIAL = "Specialized Advice - Financial"

    # PII Vulnerabilties
    PII_API_DB = "API and Database Access"
    PII_DIRECT = "Direct PII Disclosure"
    PII_SESSION = "Session PII Leak"
    PII_SOCIAL = "Social Engineering PII Disclosure"

    # Brand Vulnerabilties
    CONTRACTS = "Contracts"
    EXCESSIVE_AGENCY = "Excessive Agency"
    HALLUCINATION = "Hallucination"
    IMITATION = "Imitation"
    POLITICS = "Political Statements"

    # Technical Vulnerabilties
    DEBUG_ACCESS = "Debug Access"
    RBAC = "Role-Based Access Control"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"

    # Harm Vulnerabilties Promptfoo
    HARMFUL_VIOLENT_CRIME = (
        RTUnalignedVulnerabilities.HARMFUL_VIOLENT_CRIME.value
    )
    HARMFUL_NON_VIOLENT_CRIME = (
        RTUnalignedVulnerabilities.HARMFUL_NON_VIOLENT_CRIME.value
    )
    HARMFUL_SEX_CRIME = RTUnalignedVulnerabilities.HARMFUL_SEX_CRIME.value
    HARMFUL_CHILD_EXPLOITATION = (
        RTUnalignedVulnerabilities.HARMFUL_CHILD_EXPLOITATION.value
    )
    HARMFUL_INDISCRIMINATE_WEAPONS = (
        RTUnalignedVulnerabilities.HARMFUL_INDISCRIMINATE_WEAPONS.value
    )
    HARMFUL_HATE = RTUnalignedVulnerabilities.HARMFUL_HATE.value
    HARMFUL_SELF_HARM = RTUnalignedVulnerabilities.HARMFUL_SELF_HARM.value
    HARMFUL_SEXUAL_CONTENT = (
        RTUnalignedVulnerabilities.HARMFUL_SEXUAL_CONTENT.value
    )
    HARMFUL_CYBERCRIME = RTUnalignedVulnerabilities.HARMFUL_CYBERCRIME.value
    HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS = (
        RTUnalignedVulnerabilities.HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS.value
    )
    HARMFUL_ILLEGAL_DRUGS = (
        RTUnalignedVulnerabilities.HARMFUL_ILLEGAL_DRUGS.value
    )
    HARMFUL_COPYRIGHT_VIOLATIONS = (
        RTUnalignedVulnerabilities.HARMFUL_COPYRIGHT_VIOLATIONS.value
    )
    HARMFUL_HARASSMENT_BULLYING = (
        RTUnalignedVulnerabilities.HARMFUL_HARASSMENT_BULLYING.value
    )
    HARMFUL_ILLEGAL_ACTIVITIES = (
        RTUnalignedVulnerabilities.HARMFUL_ILLEGAL_ACTIVITIES.value
    )
    HARMFUL_GRAPHIC_CONTENT = (
        RTUnalignedVulnerabilities.HARMFUL_GRAPHIC_CONTENT.value
    )
    HARMFUL_UNSAFE_PRACTICES = (
        RTUnalignedVulnerabilities.HARMFUL_UNSAFE_PRACTICES.value
    )
    HARMFUL_RADICALIZATION = (
        RTUnalignedVulnerabilities.HARMFUL_RADICALIZATION.value
    )
    HARMFUL_PROFANITY = RTUnalignedVulnerabilities.HARMFUL_PROFANITY.value
    HARMFUL_INSULTS = RTUnalignedVulnerabilities.HARMFUL_INSULTS.value
