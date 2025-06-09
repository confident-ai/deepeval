from enum import Enum
from typing import Optional, List
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time
import json
import os
from pydantic import BaseModel

from deepeval.prompt.api import (
    PromptHttpResponse,
    PromptMessage,
    PromptType,
    PromptInterpolationType,
)
from deepeval.prompt.utils import interpolate_text
from deepeval.utils import is_confident
from deepeval.confident.api import Api, Endpoints, HttpMethods

from deepeval.constants import HIDDEN_DIR

CACHE_FILE_NAME = f"{HIDDEN_DIR}/.deepeval-prompt-cache.json"


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BaseModel):
            return obj.model_dump(by_alias=True, exclude_none=True)
        return json.JSONEncoder.default(self, obj)


class CachedPrompt(BaseModel):
    alias: str
    version: str
    template: Optional[str]
    messages_template: Optional[List[PromptMessage]]
    prompt_version_id: str
    type: PromptType
    interpolation_type: PromptInterpolationType

    class Config:
        use_enum_values = True


class Prompt:
    _prompt_version_id: Optional[str] = None
    _type: Optional[PromptType] = None
    _interpolation_type: Optional[PromptInterpolationType] = None

    def __init__(
        self,
        alias: Optional[str] = None,
        template: Optional[str] = None,
        messages_template: Optional[List[PromptMessage]] = None,
    ):
        if alias is None and template is None:
            raise TypeError(
                "Unable to create Prompt where 'alias' and 'template' are both None. Please provide at least one to continue."
            )

        self.alias = alias
        self._text_template = template
        self._messages_template = messages_template
        self.version = None

    def interpolate(self, **kwargs):
        if self._type == PromptType.TEXT:
            if self._text_template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI or set template manually to continue."
                )

            return interpolate_text(
                self._interpolation_type, self._text_template, **kwargs
            )

        elif self._type == PromptType.LIST:
            if self._messages_template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template messages. Please pull a prompt from Confident AI or set template manually to continue."
                )

            interpolated_messages = []
            for message in self._messages_template:
                interpolated_content = interpolate_text(
                    self._interpolation_type, message.content, **kwargs
                )
                interpolated_messages.append(
                    {"role": message.role, "content": interpolated_content}
                )
            return interpolated_messages
        else:
            raise ValueError(f"Unsupported prompt type: {self._type}")

    def _read_from_cache(
        self, alias: str, version: Optional[str] = None
    ) -> Optional[CachedPrompt]:
        if not os.path.exists(CACHE_FILE_NAME):
            raise Exception("No Prompt cache file found")

        try:
            with open(CACHE_FILE_NAME, "r") as f:
                cache_data = json.load(f)

            if alias in cache_data:
                if version:
                    if version in cache_data[alias]:
                        return CachedPrompt(**cache_data[alias][version])
                    else:
                        raise Exception(
                            f"Unable to find Prompt version: '{version}' for alias: '{alias}' in cache"
                        )
                else:
                    raise Exception(
                        f"Unable to load Prompt with alias: '{alias}' from cache when no version is specified "
                    )
            else:
                raise Exception(
                    f"Unable to find Prompt with alias: '{alias}' in cache"
                )
        except Exception as e:
            raise Exception(f"Error reading Prompt cache from disk: {e}")

    def _write_to_cache(self):
        if not self.alias or not self.version:
            return

        cache_data = {}
        if os.path.exists(CACHE_FILE_NAME):
            try:
                with open(CACHE_FILE_NAME, "r") as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}

        # Ensure the cache structure is initialized properly
        if self.alias not in cache_data:
            cache_data[self.alias] = {}

        # Cache the prompt
        cache_data[self.alias][self.version] = {
            "alias": self.alias,
            "version": self.version,
            "template": self._text_template,
            "messages_template": self._messages_template,
            "prompt_version_id": self._prompt_version_id,
            "type": self._type,
            "interpolation_type": self._interpolation_type,
        }

        # Ensure directory exists
        os.makedirs(HIDDEN_DIR, exist_ok=True)

        # Write back to cache file
        with open(CACHE_FILE_NAME, "w") as f:
            json.dump(cache_data, f, cls=CustomEncoder)

    def pull(
        self,
        version: Optional[str] = None,
        fallback_to_cache: bool = True,
        write_to_cache: bool = True,
    ):
        if self.alias is None:
            raise TypeError(
                "Unable to pull prompt from Confident AI when no alias is provided."
            )

        if is_confident():
            api = Api()
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Pulling [rgb(106,0,255)]'{self.alias}' (version='{version or 'latest'}')[/rgb(106,0,255)] from Confident AI...",
                    total=100,
                )
                start_time = time.perf_counter()
                try:
                    result = api.send_request(
                        method=HttpMethods.GET,
                        endpoint=Endpoints.PROMPT_ENDPOINT,
                        params={"alias": self.alias, "version": version},
                    )
                    response = PromptHttpResponse(
                        promptVersionId=result["promptVersionId"],
                        template=result["value"],
                        messages=result["messages"],
                        type=result["type"],
                        interpolation_type=result["interpolationType"],
                    )
                except:
                    try:
                        if fallback_to_cache:
                            cached_prompt = self._read_from_cache(
                                self.alias, version
                            )
                            if cached_prompt:
                                self.version = cached_prompt.version
                                self._text_template = cached_prompt.template
                                self._messages_template = (
                                    cached_prompt.messages_template
                                )
                                self._prompt_version_id = (
                                    cached_prompt.prompt_version_id
                                )
                                self._type = PromptType(cached_prompt.type)
                                self._interpolation_type = (
                                    PromptInterpolationType(
                                        cached_prompt.interpolation_type
                                    )
                                )

                                end_time = time.perf_counter()
                                time_taken = format(
                                    end_time - start_time, ".2f"
                                )
                                progress.update(
                                    task_id,
                                    description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Loaded from cache! ({time_taken}s)",
                                )
                                return
                    except:
                        raise

                self.version = version or "latest"
                self._text_template = response.template
                self._messages_template = response.messages
                self._prompt_version_id = response.promptVersionId
                self._type = response.type
                self._interpolation_type = response.interpolation_type

                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Done! ({time_taken}s)",
                )
                if write_to_cache:
                    self._write_to_cache()
        else:
            raise Exception(
                "Run `deepeval login` to pull prompt template from Confident AI"
            )
