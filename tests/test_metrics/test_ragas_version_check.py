import sys
import types

import pytest

from deepeval.metrics.ragas import import_ragas


def _install_fake_ragas(monkeypatch, version):
    module = types.ModuleType("ragas")
    if version is not None:
        module.__version__ = version
    monkeypatch.setitem(sys.modules, "ragas", module)
    return module


@pytest.mark.parametrize("version", ["0.2.1", "0.3.0", "0.10.0", "1.0.0"])
def test_import_ragas_accepts_supported_versions(monkeypatch, version):
    # 0.10.0 is the regression from #2905: a lexicographic compare made
    # "0.10.0" < "0.2.1" true, falsely rejecting it.
    _install_fake_ragas(monkeypatch, version)
    import_ragas()


@pytest.mark.parametrize("version", ["0.0.1", "0.2.0", "0.1.9"])
def test_import_ragas_rejects_old_versions(monkeypatch, version):
    _install_fake_ragas(monkeypatch, version)
    with pytest.raises(ImportError, match="0.2.1 or higher is required"):
        import_ragas()


def test_import_ragas_requires_version_attribute(monkeypatch):
    _install_fake_ragas(monkeypatch, None)
    with pytest.raises(
        ImportError, match="Version information is not available"
    ):
        import_ragas()
