import pytest
from deepteam.frameworks import Aegis
from deepteam import red_team


class TestAegis:

    def test_aegis_init(self):
        """Test that Aegis framework can be instantiated."""
        framework = Aegis()
        assert framework is not None

    def test_aegis_name(self):
        """Test that Aegis framework has correct name."""
        framework = Aegis()
        assert framework.name == "Aegis"

    def test_aegis_description(self):
        """Test that Aegis framework has correct description."""
        framework = Aegis()
        assert (
            framework.description
            == "A dataset-driven safety framework that uses the NVIDIA Aegis AI Content Safety Dataset — an open-source dataset aligned with NVIDIA's Content Safety Taxonomy. Tests models using real human-labeled safety violations from public conversations across 13 critical harm categories including sexual content, violence, hate speech, self-harm, misinformation, privacy violations, child exploitation, drugs, and terrorism. Enables evaluation of model refusal behaviors and fine-tuning of safety filters using authentic, labeled harm categories."
        )

    def test_aegis_red_team(self):
        """Test that Aegis framework can be used with red_team function."""

        def mock_model_callback(prompt: str) -> str:
            return (
                "I'm a helpful AI assistant. I cannot help with that request."
            )

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=Aegis(num_attacks=5),
            async_mode=False,
            ignore_errors=False,
        )

        assert risk_assessment is not None

    @pytest.mark.asyncio
    async def test_aegis_async_red_team(self):
        """Test that Aegis framework can be used with red_team function in async mode."""

        async def mock_async_model_callback(prompt: str) -> str:
            return (
                "I'm a helpful AI assistant. I cannot help with that request."
            )

        risk_assessment = red_team(
            model_callback=mock_async_model_callback,
            framework=Aegis(num_attacks=5),
            async_mode=True,
            ignore_errors=False,
        )

        assert risk_assessment is not None
