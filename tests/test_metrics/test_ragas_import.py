import sys
import types

import pytest

from deepeval.metrics.ragas import import_ragas


@pytest.fixture
def fake_ragas(monkeypatch):
    def _install(version):
        module = types.ModuleType("ragas")
        if version is not None:
            module.__version__ = version
        monkeypatch.setitem(sys.modules, "ragas", module)
        return module

    return _install


@pytest.mark.parametrize(
    "version",
    ["0.2.1", "0.2.10", "0.3.5", "0.10.0", "0.11.2", "1.0.0"],
)
def test_import_ragas_accepts_required_or_newer_versions(fake_ragas, version):
    # Regression test for #2905: the version check compared strings
    # lexicographically, so "0.10.0" < "0.2.1" evaluated to True and any
    # release with a two-digit minor version was rejected as outdated.
    fake_ragas(version)
    import_ragas()


@pytest.mark.parametrize("version", ["0.1.0", "0.1.22", "0.2.0"])
def test_import_ragas_rejects_older_versions(fake_ragas, version):
    fake_ragas(version)
    with pytest.raises(ImportError, match="0.2.1 or higher is required"):
        import_ragas()


def test_import_ragas_rejects_missing_version_attribute(fake_ragas):
    fake_ragas(None)
    with pytest.raises(
        ImportError, match="Version information is not available"
    ):
        import_ragas()


def test_import_ragas_tolerates_non_pep_440_versions(fake_ragas):
    fake_ragas("unknown")
    import_ragas()
