from deepteam.code_scanner import (
    CONFIG_FILENAMES,
    find_config_file,
    load_config,
)


class TestConfig:
    def test_loads_yaml(self, tmp_path):
        (tmp_path / CONFIG_FILENAMES[0]).write_text(
            "min_severity: high\n"
            "instruction: |\n"
            "  Be strict.\n"
            "exclude:\n"
            "  - 'tests/**'\n"
        )
        cfg = load_config(directory=str(tmp_path))
        assert cfg.min_severity == "high"
        assert cfg.instruction.strip() == "Be strict."
        assert cfg.exclude == ["tests/**"]

    def test_defaults_when_missing(self, tmp_path):
        cfg = load_config(directory=str(tmp_path))
        assert cfg.min_severity == "low"
        assert cfg.instruction is None

    def test_explicit_path(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("min_severity: critical\n")
        assert load_config(path=str(p)).min_severity == "critical"

    def test_find_config_file(self, tmp_path):
        assert find_config_file(str(tmp_path)) is None
        (tmp_path / CONFIG_FILENAMES[0]).write_text("min_severity: low\n")
        assert find_config_file(str(tmp_path)).endswith(CONFIG_FILENAMES[0])
