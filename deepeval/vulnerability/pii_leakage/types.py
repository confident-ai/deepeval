from enum import Enum


class PIILeakageType(Enum):
    DATABASE_ACCESS = "API and Database Access"
    DIRECT = "Direct PII Disclosure"
    SESSION_LEAK = "Session PII Leak"
    SOCIAL_MANIPULATION = "Social Engineering PII Disclosure"
