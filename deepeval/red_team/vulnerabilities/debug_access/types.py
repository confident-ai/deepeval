from enum import Enum


class DebugAccessType(Enum):
    """
    Enum for Debug Access vulnerability types.

    - Debug mode bypass through unauthorized access attempts
    - Development endpoint access and administrative interface exposure
    - Security boundary violations between production and development
    """

    DEBUG_MODE_BYPASS = "debug_mode_bypass"
    DEVELOPMENT_ENDPOINT_ACCESS = "development_endpoint_access"
    ADMINISTRATIVE_INTERFACE_EXPOSURE = "administrative_interface_exposure"


# List of all available types for easy access
DEBUG_ACCESS_TYPES = [
    DebugAccessType.DEBUG_MODE_BYPASS,
    DebugAccessType.DEVELOPMENT_ENDPOINT_ACCESS,
    DebugAccessType.ADMINISTRATIVE_INTERFACE_EXPOSURE,
]
