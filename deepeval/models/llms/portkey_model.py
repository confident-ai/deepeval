import aiohttp
import requests
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyUrl, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.utils import (
    require_secret_api_key,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import require_param


def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


class PortkeyModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[AnyUrl] = None,
        provider: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model = model or settings.PORTKEY_MODEL_NAME

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.PORTKEY_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.PORTKEY_BASE_URL is not None:
            base_url = str(settings.PORTKEY_BASE_URL).rstrip("/")

        provider = provider or settings.PORTKEY_PROVIDER_NAME

        # validation
        model = require_param(
            model,
            provider_label="Portkey",
            env_var_name="PORTKEY_MODEL_NAME",
            param_hint="model",
        )

        self.base_url = require_param(
            base_url,
            provider_label="Portkey",
            env_var_name="PORTKEY_BASE_URL",
            param_hint="base_url",
        )

        self.provider = require_param(
            provider,
            provider_label="Portkey",
            env_var_name="PORTKEY_PROVIDER_NAME",
            param_hint="provider",
        )
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    def _headers(self) -> Dict[str, str]:
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Portkey",
            env_var_name="PORTKEY_API_KEY",
            param_hint="`api_key` to PortkeyModel(...)",
        )

        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": api_key,
        }
        if self.provider:
            headers["x-portkey-provider"] = self.provider
        return headers

    def _payload(self, prompt: str) -> Dict[str, Any]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
        }
        if self.generation_kwargs:
            payload.update(self.generation_kwargs)
        return payload

    def generate_content(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": element.url},
                        }
                    )
                else:
                    element.ensure_images_loaded()
                    data_uri = (
                        f"data:{element.mimeType};base64,{element.dataBase64}"
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    )
        return content

    def _extract_content(self, data: Dict[str, Any]) -> str:
        choices: Union[List[Dict[str, Any]], None] = data.get("choices")
        if not choices:
            raise DeepEvalError("Portkey response did not include any choices.")
        message = choices[0].get("message", {})
        content: Union[str, List[Dict[str, Any]], None] = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content)
        return ""

    def generate(self, prompt: str) -> str:

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(prompt),
                headers=self._headers(),
                timeout=_request_timeout_seconds(),
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            body: Union[str, Dict[str, Any]]
            try:
                body = response.json()
            except Exception:
                body = response.text
            raise DeepEvalError(
                f"Portkey request failed with status {response.status_code}: {body}"
            ) from error
        except requests.RequestException as error:
            raise DeepEvalError(f"Portkey request failed: {error}") from error
        return self._extract_content(response.json())

    async def a_generate(self, prompt: str) -> str:

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(prompt),
                headers=self._headers(),
                timeout=_request_timeout_seconds(),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise DeepEvalError(
                        f"Portkey request failed with status {response.status}: {body}"
                    )
                data = await response.json()
                return self._extract_content(data)

    def load_model(self):
        return None

    def get_model_name(self):
        return f"{self.name} (Portkey)"

    def supports_multimodal(self):
        return True
