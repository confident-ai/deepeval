import aiohttp
import requests
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyUrl, SecretStr
import base64
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

        self.name = require_param(
            model,
            provider_label="Portkey",
            env_var_name="PORTKEY_MODEL_NAME",
            param_hint="model",
        )

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = settings.PORTKEY_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.PORTKEY_BASE_URL is not None:
            base_url = str(settings.PORTKEY_BASE_URL).rstrip("/")

        self.base_url = require_param(
            base_url,
            provider_label="Portkey",
            env_var_name="PORTKEY_BASE_URL",
            param_hint="base_url",
        )

        provider = provider or settings.PORTKEY_PROVIDER_NAME
        self.provider = require_param(
            provider,
            provider_label="Portkey",
            env_var_name="PORTKEY_PROVIDER_NAME",
            param_hint="provider",
        )
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}

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
            content = self.generate_payload_kimi(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
        }
        if self.generation_kwargs:
            payload.update(self.generation_kwargs)
        return payload

    def generate_payload_portkey(self, multimodal_input):
        """
        Converts multimodal input into Kimi/OpenAI-compatible messages.
        """
        content = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                content.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                mime, raw_bytes = self._parse_image(ele)
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
                content.append(
                    {"type": "image", "image_url": f"data:{mime};base64,{b64}"}
                )
        return content

    def _parse_image(self, image: MLLMImage):
        if image.dataBase64:
            mime = image.mimeType or "image/jpeg"
            return mime, base64.b64decode(image.dataBase64)

        if image.local and image.filename:
            from PIL import Image
            from io import BytesIO

            img = Image.open(image.filename)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            fmt = (image.mimeType or "image/jpeg").split("/")[-1].lower()
            fmt = "jpeg" if fmt == "jpg" else fmt
            img.save(buf, format=fmt.upper())
            return f"image/{fmt}", buf.getvalue()

        if image.url:
            import requests

            resp = requests.get(image.url)
            resp.raise_for_status()
            mime = resp.headers.get(
                "content-type", image.mimeType or "image/jpeg"
            )
            return mime, resp.content

        raise ValueError(
            "MLLMImage must contain dataBase64, or (local=True + filename), or url."
        )

    def _extract_content(self, data: Dict[str, Any]) -> str:
        choices: Union[List[Dict[str, Any]], None] = data.get("choices")
        if not choices:
            raise ValueError("Portkey response did not include any choices.")
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
                timeout=60,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            body: Union[str, Dict[str, Any]]
            try:
                body = response.json()
            except Exception:
                body = response.text
            raise ValueError(
                f"Portkey request failed with status {response.status_code}: {body}"
            ) from error
        except requests.RequestException as error:
            raise ValueError(f"Portkey request failed: {error}") from error
        return self._extract_content(response.json())

    async def a_generate(self, prompt: str) -> str:

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(prompt),
                headers=self._headers(),
                timeout=60,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise ValueError(
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
