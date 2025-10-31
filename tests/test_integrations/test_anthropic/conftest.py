import pytest


@pytest.fixture(scope="function", autouse=True)
def _setup_anthropic_instrumentation():
    from deepeval.anthropic.patch import (
        # patch_anthropic_classes,
        # unpatch_anthropic_classes,
        _ANTHROPIC_PATCHED,
    )

    # patch_anthropic_classes()
    # yield
    # unpatch_anthropic_classes()
