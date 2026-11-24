import pytest
from deepteam.frameworks import BeaverTails
from deepteam import red_team


class TestBeaverTails:

    def test_beavertails_init(self):
        """Test that BeaverTails framework can be instantiated."""
        framework = BeaverTails()
        assert framework is not None

    def test_beavertails_name(self):
        """Test that BeaverTails framework has correct name."""
        framework = BeaverTails()
        assert framework.name == "BeaverTails"

    def test_beavertails_description(self):
        """Test that BeaverTails framework has correct description."""
        framework = BeaverTails()
        assert (
            framework.description
            == "A dataset-driven safety framework that uses the BeaverTails dataset — a large, human-labeled collection of harmful and borderline prompts. Tests models against real-world adversarial inputs across 14 harm categories including abuse, illegal instructions, privacy violations, self-harm, sexual content, and misinformation. Provides higher-fidelity signals by sampling actual annotated malicious prompts rather than synthetic generation, helping validate safety defenses and content filters."
        )

    def test_beavertails_red_team(self):
        """Test that BeaverTails framework can be used with red_team function."""

        def mock_model_callback(prompt: str) -> str:
            return (
                "I'm a helpful AI assistant. I cannot help with that request."
            )

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=BeaverTails(num_attacks=5),
            async_mode=False,
            ignore_errors=False,
        )

        assert risk_assessment is not None

    @pytest.mark.asyncio
    async def test_beavertails_async_red_team(self):
        """Test that BeaverTails framework can be used with red_team function in async mode."""

        async def mock_async_model_callback(prompt: str) -> str:
            return (
                "I'm a helpful AI assistant. I cannot help with that request."
            )

        risk_assessment = red_team(
            model_callback=mock_async_model_callback,
            framework=BeaverTails(num_attacks=5),
            async_mode=True,
            ignore_errors=False,
        )

        assert risk_assessment is not None
