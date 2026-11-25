from enum import Enum


class SQLInjectionType(Enum):
    """
    Enum for SQL Injection vulnerability types.

    - Blind SQL injection through inference-based attacks
    - Union-based injection for data extraction
    - Error-based injection exploiting database error messages
    """

    BLIND_SQL_INJECTION = "blind_sql_injection"
    UNION_BASED_INJECTION = "union_based_injection"
    ERROR_BASED_INJECTION = "error_based_injection"


# List of all available types for easy access
SQL_INJECTION_TYPES = [
    SQLInjectionType.BLIND_SQL_INJECTION,
    SQLInjectionType.UNION_BASED_INJECTION,
    SQLInjectionType.ERROR_BASED_INJECTION,
]
