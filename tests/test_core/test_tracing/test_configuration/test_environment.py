import pytest
from deepeval.tracing.tracing import TraceManager
from deepeval.tracing.utils import Environment


class TestEnvironmentInit:
    """Tests for environment setting on TraceManager initialization."""

    def test_default_environment_is_development(self, monkeypatch):
        """Test that default environment is 'development' when no env var set."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.environment == Environment.DEVELOPMENT.value

    def test_init_with_production_env_var(self, monkeypatch):
        """Test initialization with CONFIDENT_TRACE_ENVIRONMENT=production."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "production")
        manager = TraceManager()
        assert manager.environment == "production"

    def test_init_with_staging_env_var(self, monkeypatch):
        """Test initialization with CONFIDENT_TRACE_ENVIRONMENT=staging."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "staging")
        manager = TraceManager()
        assert manager.environment == "staging"

    def test_init_with_testing_env_var(self, monkeypatch):
        """Test initialization with CONFIDENT_TRACE_ENVIRONMENT=testing."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "testing")
        manager = TraceManager()
        assert manager.environment == "testing"

    def test_init_with_invalid_env_var_raises(self, monkeypatch):
        """Test that invalid environment raises ValueError on init."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "invalid_env")
        with pytest.raises(ValueError, match="Invalid environment"):
            TraceManager()


class TestEnvironmentConfigure:
    """Tests for environment setting via configure()."""

    def test_configure_production(self, monkeypatch):
        """Test configuring environment to production."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.environment == "development"

        manager.configure(environment="production")
        assert manager.environment == "production"

    def test_configure_staging(self, monkeypatch):
        """Test configuring environment to staging."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        manager.configure(environment="staging")
        assert manager.environment == "staging"

    def test_configure_testing(self, monkeypatch):
        """Test configuring environment to testing."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        manager.configure(environment="testing")
        assert manager.environment == "testing"

    def test_configure_development(self, monkeypatch):
        """Test configuring environment to development."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "production")
        manager = TraceManager()
        assert manager.environment == "production"

        manager.configure(environment="development")
        assert manager.environment == "development"

    def test_configure_invalid_environment_raises(self, monkeypatch):
        """Test that invalid environment raises ValueError on configure."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        with pytest.raises(ValueError, match="Invalid environment"):
            manager.configure(environment="invalid")

    def test_configure_none_does_not_change(self, monkeypatch):
        """Test that configure(environment=None) doesn't change the value."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", "production")
        manager = TraceManager()
        assert manager.environment == "production"

        manager.configure(environment=None)
        assert manager.environment == "production"

    def test_configure_environment_case_sensitive(self, monkeypatch):
        """Test that environment values are case-sensitive."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        with pytest.raises(ValueError, match="Invalid environment"):
            manager.configure(environment="Production")  # Wrong case

        with pytest.raises(ValueError, match="Invalid environment"):
            manager.configure(environment="PRODUCTION")  # Wrong case


class TestAllEnvironmentValues:
    """Test all valid environment values."""

    @pytest.mark.parametrize(
        "env_value",
        [
            "production",
            "development",
            "staging",
            "testing",
        ],
    )
    def test_all_valid_environments_init(self, monkeypatch, env_value):
        """Test all valid environment values on init."""
        monkeypatch.setenv("CONFIDENT_TRACE_ENVIRONMENT", env_value)
        manager = TraceManager()
        assert manager.environment == env_value

    @pytest.mark.parametrize(
        "env_value",
        [
            "production",
            "development",
            "staging",
            "testing",
        ],
    )
    def test_all_valid_environments_configure(self, monkeypatch, env_value):
        """Test all valid environment values via configure."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment=env_value)
        assert manager.environment == env_value
