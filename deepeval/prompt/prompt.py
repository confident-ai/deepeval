from enum import Enum
from typing import Optional, List, Dict, Type
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
import time
import json
import os
from pydantic import BaseModel, ValidationError
import asyncio

from deepeval.prompt.api import (
    PromptHttpResponse,
    PromptMessage,
    PromptType,
    PromptInterpolationType,
    PromptPushRequest,
    PromptVersionsHttpResponse,
    PromptMessageList,
    PromptUpdateRequest,
    ModelSettings,
    OutputSchema,
    OutputType,
)
from deepeval.prompt.utils import (
    interpolate_text,
    construct_base_model,
    construct_output_schema,
)
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.constants import HIDDEN_DIR
from deepeval.utils import get_or_create_general_event_loop

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
    model_settings: Optional[ModelSettings]
    output_type: Optional[OutputType]
    output_schema: Optional[OutputSchema]

    class Config:
        use_enum_values = True


class Prompt:

    def __init__(
        self,
        alias: Optional[str] = None,
        text_template: Optional[str] = None,
        messages_template: Optional[List[PromptMessage]] = None,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        interpolation_type: Optional[PromptInterpolationType] = None,
    ):
        if text_template and messages_template:
            raise TypeError(
                "Unable to create Prompt where 'text_template' and 'messages_template' are both provided. Please provide only one to continue."
            )
        self.alias = alias
        self.text_template = text_template
        self.messages_template = messages_template
        self.model_settings: Optional[ModelSettings] = model_settings
        self.output_type: Optional[OutputType] = output_type
        self.output_schema: Optional[Type[BaseModel]] = output_schema
        self.interpolation_type: Optional[PromptInterpolationType] = interpolation_type

        self.type: Optional[PromptType] = None
        if text_template:
            self.type = PromptType.TEXT
        elif messages_template:
            self.type = PromptType.LIST

        self._version = None
        self._prompt_version_id: Optional[str] = None
        self._polling_tasks: Dict[str, asyncio.Task] = {}
        self._refresh_map: Dict[str, int] = {}

    @property
    def version(self):
        if self._version is not None and self._version != "latest":
            return self._version
        versions = self._get_versions()
        if len(versions) == 0:
            return "latest"
        else:
            return versions[-1].version

    @version.setter
    def version(self, value):
        self._version = value

    def load(self, file_path: str, messages_key: Optional[str] = None):
        _, ext = os.path.splitext(file_path)
        if ext != ".json" and ext != ".txt":
            raise ValueError("Only .json and .txt files are supported")

        file_name = os.path.basename(file_path).split(".")[0]
        self.alias = file_name
        with open(file_path, "r") as f:
            content = f.read()
        try:
            data = json.loads(content)
        except:
            self.template = content
            return content

        text_template = None
        messages_template = None
        try:
            if isinstance(data, list):
                messages_template = PromptMessageList.validate_python(data)
            elif isinstance(data, dict):
                if messages_key is None:
                    raise ValueError(
                        "messages `key` must be provided if file is a dictionary"
                    )
                messages = data[messages_key]
                messages_template = PromptMessageList.validate_python(messages)
            else:
                text_template = content
        except ValidationError:
            text_template = content

        self.template = text_template
        self.messages_template = messages_template
        return text_template or messages_template

    def interpolate(self, **kwargs):
        if self.type == PromptType.TEXT:
            if self.template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI or set template manually to continue."
                )

            return interpolate_text(
                self.interpolation_type, self.text_template, **kwargs
            )

        elif self.type == PromptType.LIST:
            if self.messages_template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template messages. Please pull a prompt from Confident AI or set template manually to continue."
                )

            interpolated_messages = []
            for message in self.messages_template:
                interpolated_content = interpolate_text(
                    self.interpolation_type, message.content, **kwargs
                )
                interpolated_messages.append(
                    {"role": message.role, "content": interpolated_content}
                )
            return interpolated_messages
        else:
            raise ValueError(f"Unsupported prompt type: {self.type}")

    ############################################
    ### Pull, Push, Update
    ############################################

    def pull(
        self,
        version: Optional[str] = None,
        fallback_to_cache: bool = True,
        write_to_cache: bool = True,
        default_to_cache: bool = True,
        refresh: Optional[int] = 60,
    ):
        if refresh:
            default_to_cache = True
            write_to_cache = False
        if self.alias is None:
            raise TypeError(
                "Unable to pull prompt from Confident AI when no alias is provided."
            )

        # Manage background prompt polling
        if refresh:
            loop = get_or_create_general_event_loop()
            if loop.is_running():
                loop.create_task(self.create_polling_task(version, refresh))
            else:
                loop.run_until_complete(
                    self.create_polling_task(version, refresh)
                )

        if default_to_cache:
            try:
                cached_prompt = self._read_from_cache(self.alias, version)
                if cached_prompt:
                    self._version = cached_prompt.version
                    self.text_template = cached_prompt.template
                    self.messages_template = cached_prompt.messages_template
                    self._prompt_version_id = cached_prompt.prompt_version_id
                    self.type = PromptType(cached_prompt.type)
                    self.interpolation_type = PromptInterpolationType(
                        cached_prompt.interpolation_type
                    )
                    self.model_settings = cached_prompt.model_settings
                    self.output_type = OutputType(cached_prompt.output_type)
                    self.output_schema = construct_base_model(
                        cached_prompt.output_schema
                    )
                    return
            except:
                pass

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
                data, _ = api.send_request(
                    method=HttpMethods.GET,
                    endpoint=Endpoints.PROMPTS_VERSION_ID_ENDPOINT,
                    url_params={
                        "alias": self.alias,
                        "versionId": version or "latest",
                    },
                )
                response = PromptHttpResponse(
                    id=data["id"],
                    text=data.get("text", None),
                    messages=data.get("messages", None),
                    type=data["type"],
                    interpolation_type=data["interpolationType"],
                    model_settings=data.get("modelSettings", None),
                    output_type=data.get("outputType", None),
                    output_schema=data.get("outputSchema", None),
                )
            except:
                try:
                    if fallback_to_cache:
                        cached_prompt = self._read_from_cache(
                            self.alias, version
                        )
                        if cached_prompt:
                            self._version = cached_prompt.version
                            self.text_template = cached_prompt.template
                            self.messages_template = (
                                cached_prompt.messages_template
                            )
                            self._prompt_version_id = (
                                cached_prompt.prompt_version_id
                            )
                            self.type = PromptType(cached_prompt.type)
                            self.interpolation_type = PromptInterpolationType(
                                cached_prompt.interpolation_type
                            )
                            self.model_settings = cached_prompt.model_settings
                            self.output_type = OutputType(
                                cached_prompt.output_type
                            )
                            self.output_schema = construct_base_model(
                                cached_prompt.output_schema
                            )

                            end_time = time.perf_counter()
                            time_taken = format(end_time - start_time, ".2f")
                            progress.update(
                                task_id,
                                description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Loaded from cache! ({time_taken}s)",
                            )
                            return
                except:
                    raise

            self._version = version or "latest"
            self.text_template = response.text
            self.messages_template = response.messages
            self._prompt_version_id = response.id
            self.type = response.type
            self.interpolation_type = response.interpolation_type
            self.model_settings = response.model_settings
            self.output_type = response.output_type
            self.output_schema = construct_base_model(response.output_schema)

            end_time = time.perf_counter()
            time_taken = format(end_time - start_time, ".2f")
            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Done! ({time_taken}s)",
            )
            if write_to_cache:
                self._write_to_cache(
                    version=version or "latest",
                    text_template=response.text,
                    messages_template=response.messages,
                    prompt_version_id=response.id,
                    type=response.type,
                    interpolation_type=response.interpolation_type,
                    model_settings=response.model_settings,
                    output_type=response.output_type,
                    output_schema=response.output_schema,
                )

    def push(
        self,
        text: Optional[str] = None,
        messages: Optional[List[PromptMessage]] = None,
        interpolation_type: Optional[
            PromptInterpolationType
        ] = PromptInterpolationType.FSTRING,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        if self.alias is None:
            raise ValueError(
                "Prompt alias is not set. Please set an alias to continue."
            )
        if text is None and messages is None:
            raise ValueError("Either text or messages must be provided")
        if text is not None and messages is not None:
            raise ValueError("Only one of text or messages can be provided")

        body = PromptPushRequest(
            alias=self.alias,
            text=text,
            messages=messages,
            interpolation_type=interpolation_type,
            model_settings=model_settings,
            output_type=output_type,
            output_schema=construct_output_schema(output_schema),
        )
        try:
            body = body.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = body.dict(by_alias=True, exclude_none=True)

        api = Api()
        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROMPTS_ENDPOINT,
            body=body,
        )
        if link:
            self.text_template = text
            self.messages_template = messages
            self.interpolation_type = interpolation_type
            self.model_settings = model_settings
            self.output_type = output_type
            self.output_schema = output_schema
            self.type = PromptType.TEXT if text else PromptType.LIST
            console = Console()
            console.print(
                "✅ Prompt successfully pushed to Confident AI! View at "
                f"[link={link}]{link}[/link]"
            )

    def update(
        self,
        version: str,
        text: Optional[str] = None,
        messages: Optional[List[PromptMessage]] = None,
        interpolation_type: Optional[
            PromptInterpolationType
        ] = PromptInterpolationType.FSTRING,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        if self.alias is None:
            raise ValueError(
                "Prompt alias is not set. Please set an alias to continue."
            )

        body = PromptUpdateRequest(
            text=text,
            messages=messages,
            interpolation_type=interpolation_type,
            model_settings=model_settings,
            output_type=output_type,
            output_schema=construct_output_schema(output_schema),
        )
        try:
            body = body.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            body = body.dict(by_alias=True, exclude_none=True)
        api = Api()
        data, _ = api.send_request(
            method=HttpMethods.PUT,
            endpoint=Endpoints.PROMPTS_VERSION_ID_ENDPOINT,
            url_params={
                "alias": self.alias,
                "versionId": version,
            },
            body=body,
        )
        if data:
            self._prompt_version_id = data["id"]
            self._version = version
            self.text_template = text
            self.messages_template = messages
            self.type = PromptType.TEXT
            self.interpolation_type = interpolation_type
            self.model_settings = model_settings
            self.output_type = output_type
            self.output_schema = output_schema
            self.type = PromptType.TEXT if text else PromptType.LIST
            console = Console()
            console.print("✅ Prompt successfully updated on Confident AI!")

    ############################################
    ### Polling
    ############################################

    async def create_polling_task(
        self,
        version: Optional[str],
        refresh: Optional[int] = 60,
    ):
        if version is None:
            return

        # If polling task doesn't exist, start it
        polling_task: Optional[asyncio.Task] = self._polling_tasks.get(version)
        if refresh:
            self._refresh_map[version] = refresh
            if not polling_task:
                self._polling_tasks[version] = asyncio.create_task(
                    self.poll(version)
                )

        # If invalid `refresh`, stop the task
        else:
            if polling_task:
                polling_task.cancel()
            self._polling_tasks.pop(version)
            self._refresh_map.pop(version)

    async def poll(self, version: Optional[str] = None):
        api = Api()
        while True:
            try:
                data, _ = api.send_request(
                    method=HttpMethods.GET,
                    endpoint=Endpoints.PROMPTS_VERSION_ID_ENDPOINT,
                    url_params={
                        "alias": self.alias,
                        "versionId": version or "latest",
                    },
                )
                response = PromptHttpResponse(
                    id=data["id"],
                    text=data.get("text", None),
                    messages=data.get("messages", None),
                    type=data["type"],
                    interpolation_type=data["interpolationType"],
                )
                self._write_to_cache(
                    version=version or "latest",
                    text_template=response.text,
                    messages_template=response.messages,
                    prompt_version_id=response.id,
                    type=response.type,
                    interpolation_type=response.interpolation_type,
                )
            except Exception as e:
                pass

            await asyncio.sleep(self._refresh_map[version])

    ############################################
    ### Utils
    ############################################

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

    def _write_to_cache(
        self,
        version: Optional[str] = None,
        text_template: Optional[str] = None,
        messages_template: Optional[List[PromptMessage]] = None,
        prompt_version_id: Optional[str] = None,
        type: Optional[PromptType] = None,
        interpolation_type: Optional[PromptInterpolationType] = None,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[OutputSchema] = None,
    ):
        if not self.alias or not version:
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
        cache_data[self.alias][version] = {
            "alias": self.alias,
            "version": version,
            "template": text_template,
            "messages_template": messages_template,
            "prompt_version_id": prompt_version_id,
            "type": type,
            "interpolation_type": interpolation_type,
            "model_settings": model_settings,
            "output_type": output_type,
            "output_schema": output_schema,
        }

        # Ensure directory exists
        os.makedirs(HIDDEN_DIR, exist_ok=True)

        # Write back to cache file
        with open(CACHE_FILE_NAME, "w") as f:
            json.dump(cache_data, f, cls=CustomEncoder)

    def _get_versions(self) -> List:
        if self.alias is None:
            raise ValueError(
                "Prompt alias is not set. Please set an alias to continue."
            )
        api = Api()
        data, _ = api.send_request(
            method=HttpMethods.GET,
            endpoint=Endpoints.PROMPTS_VERSIONS_ENDPOINT,
            url_params={"alias": self.alias},
        )
        versions = PromptVersionsHttpResponse(**data)
        return versions.text_versions or versions.messages_versions or []
