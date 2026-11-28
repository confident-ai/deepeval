from enum import Enum


class BOLAType(Enum):
    """
    Enum for BOLA (Broken Object Level Authorization) vulnerability types.

    - Object access bypass through unauthorized access attempts
    - Cross-customer access patterns with agentic data exfiltration
    - Unauthorized object manipulation and data access
    """

    OBJECT_ACCESS_BYPASS = "object_access_bypass"
    CROSS_CUSTOMER_ACCESS = "cross_customer_access"
    UNAUTHORIZED_OBJECT_MANIPULATION = "unauthorized_object_manipulation"


# List of all available types for easy access
BOLA_TYPES = [
    BOLAType.OBJECT_ACCESS_BYPASS,
    BOLAType.CROSS_CUSTOMER_ACCESS,
    BOLAType.UNAUTHORIZED_OBJECT_MANIPULATION,
]
