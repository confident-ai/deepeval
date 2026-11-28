from enum import Enum


class SSRFType(Enum):
    """
    Enum for SSRF (Server-Side Request Forgery) vulnerability types.

    - Internal service access through unauthorized server-side requests
    - Cloud metadata access for infrastructure information exposure
    - Port scanning and network reconnaissance attacks
    """

    INTERNAL_SERVICE_ACCESS = "internal_service_access"
    CLOUD_METADATA_ACCESS = "cloud_metadata_access"
    PORT_SCANNING = "port_scanning"


# List of all available types for easy access
SSRF_TYPES = [
    SSRFType.INTERNAL_SERVICE_ACCESS,
    SSRFType.CLOUD_METADATA_ACCESS,
    SSRFType.PORT_SCANNING,
]
