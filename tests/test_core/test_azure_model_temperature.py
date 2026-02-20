import pytest

from deepeval.models.llms.azure_model import AzureOpenAIModel


# Shared dummy credentials so __init__ does not raise on missing params.
_AZURE_KWARGS = dict(
    api_key="fake-key",
    base_url="https://fake.openai.azure.com",
    deployment_name="fake-deployment",
    api_version="2024-02-01",
)


class TestAzureModelTemperature:
    def test_reasoning_model_temperature_is_none(self):
        """o3-mini has supports_temperature=False; temperature must be None."""
        model = AzureOpenAIModel(model="o3-mini", **_AZURE_KWARGS)
        assert model.temperature is None

    def test_standard_model_temperature_is_set(self):
        """gpt-4o supports temperature; it should default to 0.0."""
        model = AzureOpenAIModel(model="gpt-4o", **_AZURE_KWARGS)
        assert model.temperature is not None
        assert model.temperature == 0.0

    def test_explicit_temperature_preserved_for_standard_model(self):
        """User-supplied temperature is kept for models that support it."""
        model = AzureOpenAIModel(
            model="gpt-4o", temperature=0.7, **_AZURE_KWARGS
        )
        assert model.temperature == pytest.approx(0.7)

    def test_explicit_temperature_overridden_for_reasoning_model(self):
        """Even if user passes temperature, reasoning models get None."""
        model = AzureOpenAIModel(
            model="o3-mini", temperature=0.5, **_AZURE_KWARGS
        )
        assert model.temperature is None
