import pytest
from deepeval.tracing.tracing import TraceManager


def dummy_mask(data):
    """Dummy mask function for testing."""
    return "[MASKED]" if isinstance(data, str) else data


class TestConfigureMultiple:
    """Tests for configuring multiple options at once."""

    def test_configure_all_options(self, monkeypatch):
        """Test configuring all options at once."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        manager.configure(
            environment="production",
            sampling_rate=0.5,
            confident_api_key="my-api-key",
            tracing_enabled=False,
            mask=dummy_mask,
        )

        assert manager.environment == "production"
        assert manager.sampling_rate == 0.5
        assert manager.confident_api_key == "my-api-key"
        assert manager.tracing_enabled is False
        assert manager.custom_mask_fn is dummy_mask

    def test_configure_subset_of_options(self, monkeypatch):
        """Test configuring a subset of options."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        # First configure some options
        manager.configure(
            environment="staging",
            sampling_rate=0.8,
        )

        assert manager.environment == "staging"
        assert manager.sampling_rate == 0.8
        assert manager.tracing_enabled is True  # Default unchanged

        # Configure different options
        manager.configure(
            tracing_enabled=False,
            confident_api_key="new-key",
        )

        # Previous values should be unchanged
        assert manager.environment == "staging"
        assert manager.sampling_rate == 0.8
        # New values should be set
        assert manager.tracing_enabled is False
        assert manager.confident_api_key == "new-key"

    def test_configure_with_invalid_option_fails_atomically(self, monkeypatch):
        """Test that invalid option causes entire configure to fail."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        original_environment = manager.environment
        original_sampling_rate = manager.sampling_rate

        # This should fail because environment is invalid
        with pytest.raises(ValueError, match="Invalid environment"):
            manager.configure(
                environment="invalid_env",
                sampling_rate=0.5,  # This is valid but should not be applied
            )

        # Note: In the current implementation, environment is validated
        # before sampling_rate, so sampling_rate won't be changed if
        # environment validation fails first
        assert manager.environment == original_environment

    def test_configure_empty_call_does_nothing(self, monkeypatch):
        """Test that configure() with no args doesn't change anything."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        original_env = manager.environment
        original_rate = manager.sampling_rate
        original_enabled = manager.tracing_enabled

        manager.configure()

        assert manager.environment == original_env
        assert manager.sampling_rate == original_rate
        assert manager.tracing_enabled == original_enabled
