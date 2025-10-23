import pytest


@pytest.fixture(scope="function", autouse=True)
def _setup_openai_instrumentation():
    from deepeval.openai.patch import (
        patch_openai_classes,
        unpatch_openai_classes,
    )

    patch_openai_classes()
    yield
    unpatch_openai_classes()
