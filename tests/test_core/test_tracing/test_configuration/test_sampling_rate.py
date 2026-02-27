import pytest
from deepeval.tracing.tracing import TraceManager


class TestSamplingRateInit:
    """Tests for sampling rate setting on TraceManager initialization."""

    def test_default_sampling_rate_is_one(self, monkeypatch):
        """Test that default sampling rate is 1.0 when no env var set."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.sampling_rate == 1.0

    def test_init_with_sampling_rate_env_var(self, monkeypatch):
        """Test initialization with CONFIDENT_TRACE_SAMPLE_RATE."""
        monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", "0.5")
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.sampling_rate == 0.5

    def test_init_with_zero_sampling_rate(self, monkeypatch):
        """Test initialization with sampling rate of 0."""
        monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", "0")
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.sampling_rate == 0.0

    def test_init_with_one_sampling_rate(self, monkeypatch):
        """Test initialization with sampling rate of 1."""
        monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", "1")
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.sampling_rate == 1.0


class TestSamplingRateConfigure:
    """Tests for sampling rate setting via configure()."""

    def test_configure_sampling_rate(self, monkeypatch):
        """Test configuring sampling rate."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.sampling_rate == 1.0

        manager.configure(sampling_rate=0.5)
        assert manager.sampling_rate == 0.5

    def test_configure_sampling_rate_zero(self, monkeypatch):
        """Test configuring sampling rate to 0."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        manager.configure(sampling_rate=0.0)
        assert manager.sampling_rate == 0.0

    def test_configure_sampling_rate_one(self, monkeypatch):
        """Test configuring sampling rate to 1."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(sampling_rate=0.5)

        manager.configure(sampling_rate=1.0)
        assert manager.sampling_rate == 1.0

    def test_configure_invalid_sampling_rate_negative_raises(self, monkeypatch):
        """Test that negative sampling rate raises ValueError."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        with pytest.raises(ValueError, match="Invalid sampling rate"):
            manager.configure(sampling_rate=-0.1)

    def test_configure_invalid_sampling_rate_above_one_raises(
        self, monkeypatch
    ):
        """Test that sampling rate > 1 raises ValueError."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        with pytest.raises(ValueError, match="Invalid sampling rate"):
            manager.configure(sampling_rate=1.1)

    def test_configure_none_does_not_change(self, monkeypatch):
        """Test that configure(sampling_rate=None) doesn't change the value."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(sampling_rate=0.5)

        manager.configure(sampling_rate=None)
        assert manager.sampling_rate == 0.5


class TestSamplingRateEdgeCases:
    """Test edge cases for sampling rate."""

    @pytest.mark.parametrize("rate", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_valid_sampling_rates(self, monkeypatch, rate):
        """Test various valid sampling rate values."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(sampling_rate=rate)
        assert manager.sampling_rate == rate

    @pytest.mark.parametrize("rate", [-1.0, -0.001, 1.001, 2.0, 100.0])
    def test_invalid_sampling_rates(self, monkeypatch, rate):
        """Test invalid sampling rate values."""
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        with pytest.raises(ValueError, match="Invalid sampling rate"):
            manager.configure(sampling_rate=rate)
