import pytest
from deepeval.tracing.tracing import TraceManager
from deepeval.tracing.types import EvalMode, EvalSession
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


class TestEnvironmentNotLeakedByEvaluation:
    """A finished evaluation trace must not rewrite the manager's environment.

    ``TraceManager.end_trace`` takes an ``else`` branch for the synchronous
    evaluation modes (``EVALUATE`` / ``ITERATOR_SYNC``). That branch used to
    overwrite the shared ``self.environment`` with ``Environment.TESTING`` and
    never restore it, so the value leaked into every later trace and into every
    integration that reads ``trace_manager.environment``.
    """

    @staticmethod
    def _finish_sync_eval_trace(manager):
        """Drive the ``else`` branch: finish a trace under ITERATOR_SYNC."""
        manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_SYNC)
        trace = manager.start_new_trace()
        manager.end_trace(trace.uuid)
        return trace

    def test_end_trace_keeps_configured_environment(self, monkeypatch):
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment="production")

        self._finish_sync_eval_trace(manager)

        assert manager.environment == "production"

    def test_environment_does_not_depend_on_session_reset(self, monkeypatch):
        """The environment must survive an evaluation without being repaired
        by the pipeline's ``eval_session`` reset on exit."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment="staging")

        self._finish_sync_eval_trace(manager)
        manager.eval_session = EvalSession()

        assert manager.environment == "staging"

    def test_later_trace_reports_configured_environment(self, monkeypatch):
        """A trace created after a sync evaluation reports the configured
        environment, not the leaked ``testing`` value."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment="production")

        self._finish_sync_eval_trace(manager)
        manager.eval_session = EvalSession()

        later_trace = manager.start_new_trace()
        api = manager.create_trace_api(later_trace)

        assert api.environment == "production"

    def test_global_manager_environment_not_corrupted(self, monkeypatch):
        """The process-wide singleton must stay untouched as well."""
        from deepeval.tracing.tracing import trace_manager

        original = trace_manager.environment
        monkeypatch.setattr(trace_manager, "environment", "production")
        try:
            self._finish_sync_eval_trace(trace_manager)
            assert trace_manager.environment == "production"
        finally:
            trace_manager.eval_session = EvalSession()
            trace_manager.environment = original

    def test_eval_trace_is_labeled_testing_per_trace(self, monkeypatch):
        """The evaluation trace itself keeps its ``testing`` label, while the
        manager's configured environment is left alone."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment="production")

        trace = self._finish_sync_eval_trace(manager)
        api = manager.create_trace_api(trace)

        assert api.environment == "testing"
        assert manager.environment == "production"

    def test_user_set_trace_environment_wins(self, monkeypatch):
        """A per-trace environment set by the user is not overwritten."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(environment="production")

        trace = self._finish_sync_eval_trace(manager)
        trace.environment = "staging"
        api = manager.create_trace_api(trace)

        assert api.environment == "staging"
