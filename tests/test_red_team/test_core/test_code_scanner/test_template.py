from deepteam.code_scanner import (
    DEFAULT_CODE_SCAN_VULNERABILITIES,
    CodeScanTemplate,
)
from deepteam.vulnerabilities import SSRF, BaseVulnerability


def _catalog(prompt: str) -> str:
    return prompt.split("VULNERABILITIES TO LOOK FOR")[1].split(
        "WHAT TO FOCUS ON"
    )[0]


class TestTemplate:
    def test_default_catalog_renders_known_types(self):
        p = CodeScanTemplate.generate_code_batch_evaluation(batch_data="[]")
        assert "Unexpected Code Execution: unauthorized_code_execution" in p
        assert "Prompt Leakage: secrets_and_credentials" in p

    def test_default_vulns_are_classes(self):
        assert all(
            isinstance(v, type) and issubclass(v, BaseVulnerability)
            for v in DEFAULT_CODE_SCAN_VULNERABILITIES
        )

    def test_subset_scoping_class_and_string(self):
        for vulns in ([SSRF], ["SSRF"]):
            p = CodeScanTemplate.generate_code_batch_evaluation(
                batch_data="[]", vulnerabilities=vulns
            )
            catalog = _catalog(p)
            assert "SSRF: internal_service_access" in catalog
            assert "Unexpected Code Execution:" not in catalog

    def test_instruction_injected(self):
        p = CodeScanTemplate.generate_code_batch_evaluation(
            batch_data="[]", instruction="Treat secrets as critical."
        )
        assert "ADDITIONAL PROJECT INSTRUCTION" in p
        assert "Treat secrets as critical." in p

    def test_no_instruction_no_block(self):
        p = CodeScanTemplate.generate_code_batch_evaluation(batch_data="[]")
        assert "ADDITIONAL PROJECT INSTRUCTION" not in p

    def test_batch_data_embedded(self):
        p = CodeScanTemplate.generate_code_batch_evaluation(
            batch_data='[{"x":1}]'
        )
        assert '[{"x":1}]' in p
