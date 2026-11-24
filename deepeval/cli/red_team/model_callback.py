from __future__ import annotations

from typing import Any, Dict, Union, Optional
import importlib.util
import sys
import os

from deepeval.models import (
    GPTModel,
    AzureOpenAIModel,
    OllamaModel,
    LocalModel,
    GeminiModel,
    AnthropicModel,
    AmazonBedrockModel,
    DeepEvalBaseLLM,
    # PortkeyModel,
)
from deepeval.metrics.utils import initialize_model


def _load_custom_model_from_file(
    file_path: str, class_name: str
) -> DeepEvalBaseLLM:
    """Load a custom DeepEval model class from a Python file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom model file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(
        "custom_model_module", file_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_model_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")

    model_class = getattr(module, class_name)
    if not issubclass(model_class, DeepEvalBaseLLM):
        raise TypeError(
            f"'{class_name}' in {file_path} must inherit from DeepEvalBaseLLM"
        )

    return model_class()


def load_model(spec: Union[str, Dict[str, Any], None]) -> DeepEvalBaseLLM:
    """Construct a DeepEval model instance from a YAML spec or string."""
    if spec is None or isinstance(spec, str):
        # use deepeval helper which respects global config
        model, _ = initialize_model(spec)
        return model

    provider = str(spec.get("provider", "openai")).lower()
    model_name = spec.get("model") or spec.get("model_name")
    temperature = spec.get("temperature", 0)

    if provider == "openai":
        return GPTModel(model=model_name, temperature=temperature)

    if provider == "azure":
        return AzureOpenAIModel(
            model_name=model_name,
            deployment_name=spec.get("deployment_name"),
            azure_openai_api_key=spec.get("api_key"),
            openai_api_version=spec.get("openai_api_version"),
            azure_endpoint=spec.get("endpoint") or spec.get("azure_endpoint"),
            temperature=temperature,
        )

    if provider == "ollama":
        return OllamaModel(
            model=model_name,
            base_url=spec.get("base_url", "http://localhost:11434"),
            temperature=temperature,
        )

    if provider == "local":
        return LocalModel(
            model_name=model_name,
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key"),
            temperature=temperature,
        )

    if provider == "gemini":
        if spec.get("project") and spec.get("location"):
            return GeminiModel(
                model_name=model_name,
                project=spec.get("project"),
                location=spec.get("location"),
                temperature=temperature,
            )
        return GeminiModel(
            model_name=model_name,
            api_key=spec.get("api_key"),
            temperature=temperature,
        )

    if provider == "anthropic":
        return AnthropicModel(model=model_name, temperature=temperature)

    # if provider == "portkey":
    #     return PortkeyModel(
    #         model=model_name,
    #         api_key=spec.get("api_key"),
    #         base_url=spec.get("base_url"),
    #         provider=spec.get("provider_name"),
    #     )

    if provider == "bedrock":
        return AmazonBedrockModel(
            model_id=model_name,
            region_name=spec.get("region_name"),
            aws_access_key_id=spec.get("aws_access_key_id"),
            aws_secret_access_key=spec.get("aws_secret_access_key"),
            temperature=temperature,
        )

    if provider == "custom":
        file_path = spec.get("file")
        class_name = spec.get("class")
        if not file_path or not class_name:
            raise ValueError(
                "Custom provider requires 'file' and 'class' fields"
            )
        return _load_custom_model_from_file(file_path, class_name)

    raise ValueError(f"Unknown provider: {provider}")
