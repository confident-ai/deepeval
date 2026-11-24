from typing import Dict, List, Optional, Type, Union

from deepteam.vulnerabilities.base_vulnerability import BaseVulnerability
from deepteam.vulnerabilities.constants import VULNERABILITY_INFO_MAP

VulnerabilityRef = Union[str, Type[BaseVulnerability]]

KNOWN_VULNERABILITIES: Dict[str, List[str]] = {
    name: list(info.allowed_types)
    for name, info in VULNERABILITY_INFO_MAP.items()
}


def to_name(vulnerability: VulnerabilityRef) -> str:
    """Normalize a vulnerability class/instance or name string to its name."""
    if isinstance(vulnerability, str):
        return vulnerability
    return vulnerability.name


def vulnerability_names() -> List[str]:
    """All known deepteam vulnerability category names."""
    return list(KNOWN_VULNERABILITIES.keys())


def allowed_types(vulnerability: str) -> List[str]:
    """Allowed subtypes for a category ([] if the category is unknown)."""
    return KNOWN_VULNERABILITIES.get(vulnerability, [])


def is_known(
    vulnerability: str, vulnerability_type: Optional[str] = None
) -> bool:
    """Whether a (category[, subtype]) pair exists in deepteam's taxonomy."""
    if vulnerability not in KNOWN_VULNERABILITIES:
        return False
    if vulnerability_type is None:
        return True
    return vulnerability_type in KNOWN_VULNERABILITIES[vulnerability]
