import importlib.util
import sys
import types
from pathlib import Path


def load_key_handler():
    package_name = "_key_handler_test_package"
    package_path = Path(__file__).parents[2] / "deepeval"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

    for module_name in ("constants", "key_handler"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            package_path / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.key_handler"]


def test_embedding_key_values_are_strings():
    key_handler = load_key_handler()

    assert all(
        isinstance(key.value, str) for key in key_handler.EmbeddingKeyValues
    )
