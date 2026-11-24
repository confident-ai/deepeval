from deepteam.code_scanner import (
    KNOWN_VULNERABILITIES,
    allowed_types,
    is_known,
    vulnerability_names,
)
from deepteam.code_scanner.taxonomy import to_name
from deepteam.vulnerabilities import SSRF


class TestTaxonomy:
    def test_known_populated(self):
        assert len(KNOWN_VULNERABILITIES) > 0
        assert "SSRF" in KNOWN_VULNERABILITIES

    def test_allowed_types(self):
        assert "port_scanning" in allowed_types("SSRF")
        assert allowed_types("Does Not Exist") == []

    def test_is_known(self):
        assert is_known("SSRF")
        assert is_known("SSRF", "port_scanning")
        assert not is_known("SSRF", "made_up")
        assert not is_known("Does Not Exist")

    def test_to_name_accepts_class_or_string(self):
        assert to_name(SSRF) == "SSRF"
        assert to_name("SSRF") == "SSRF"

    def test_vulnerability_names(self):
        assert "SSRF" in vulnerability_names()
