import logging
import time
import json
import os

from enum import Enum
from typing import Optional, List, Dict, Type, Literal
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
import time
import json
import os
from pydantic import BaseModel, ValidationError
import asyncio
import threading

from deepeval.utils import make_model_config, is_read_only_env

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


logger = logging.getLogger(__name__)

portalocker = None
if not is_read_only_env():
    try:
        import portalocker
    except Exception as e:
        logger.warning("failed to import portalocker: %s", e)
else:
    logger.warning("READ_ONLY filesystem: skipping disk cache for prompts.")

CACHE_FILE_NAME = f"{HIDDEN_DIR}/.deepeval-prompt-cache.json"
VERSION_CACHE_KEY = "version"
LABEL_CACHE_KEY = "label"

# Global background event loop for polling
_polling_loop: Optional[asyncio.AbstractEventLoop] = None
_polling_thread: Optional[threading.Thread] = None
_polling_loop_lock = threading.Lock()


def _get_or_create_polling_loop() -> asyncio.AbstractEventLoop:
    """Get or create a background event loop for polling that runs in a daemon thread."""
    global _polling_loop, _polling_thread

    with _polling_loop_lock:
        if _polling_loop is None or not _polling_loop.is_running():

            def run_loop():
                global _polling_loop
                _polling_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_polling_loop)
                _polling_loop.run_forever()

            _polling_thread = threading.Thread(target=run_loop, daemon=True)
            _polling_thread.start()

            # Wait for loop to be ready
            while _polling_loop is None:
                time.sleep(0.01)

        return _polling_loop


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BaseModel):
            return obj.model_dump(by_alias=True, exclude_none=True)
        return json.JSONEncoder.default(self, obj)


class CachedPrompt(BaseModel):
    model_config = make_model_config(use_enum_values=True)

    alias: str
    version: str
    label: Optional[str] = None
    template: Optional[str]
    messages_template: Optional[List[PromptMessage]]
    prompt_version_id: str
    type: PromptType
    interpolation_type: PromptInterpolationType
    model_settings: Optional[ModelSettings]
    output_type: Optional[OutputType]
    output_schema: Optional[OutputSchema]


class Prompt:

    def __init__(
        self,
        alias: Optional[str] = None,
        text_template: Optional[str] = None,
        messages_template: Optional[List[PromptMessage]] = None,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[Type[BaseModel]] = None,
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
        self.label: Optional[str] = None
        self.interpolation_type: Optional[PromptInterpolationType] = None

        self._version = None
        self._prompt_version_id: Optional[str] = None
        self._polling_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        self._refresh_map: Dict[str, Dict[str, int]] = {}
        self._lock = (
            threading.Lock()
        )  # Protect instance attributes from race conditions

        self.type: Optional[PromptType] = None
        if text_template:
            self.type = PromptType.TEXT
        elif messages_template:
            self.type = PromptType.LIST

    def __del__(self):
        """Cleanup polling tasks when instance is destroyed"""
        try:
            self._stop_polling()
        except Exception:
            # Suppress exceptions during cleanup to avoid issues in interpreter shutdown
            pass

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
        except (json.JSONDecodeError, TypeError):
            self.text_template = content
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

        self.text_template = text_template
        self.messages_template = messages_template
        return text_template or messages_template

    def interpolate(self, **kwargs):
        with self._lock:
            prompt_type = self.type
            text_template = self.text_template
            messages_template = self.messages_template
            interpolation_type = self.interpolation_type

        if prompt_type == PromptType.TEXT:
            if text_template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI or set template manually to continue."
                )

            return interpolate_text(interpolation_type, text_template, **kwargs)

        elif prompt_type == PromptType.LIST:
            if messages_template is None:
                raise TypeError(
                    "Unable to interpolate empty prompt template messages. Please pull a prompt from Confident AI or set template manually to continue."
                )

            interpolated_messages = []
            for message in messages_template:
                interpolated_content = interpolate_text(
                    interpolation_type, message.content, **kwargs
                )
                interpolated_messages.append(
                    {"role": message.role, "content": interpolated_content}
                )
            return interpolated_messages
        else:
            raise ValueError(f"Unsupported prompt type: {self.type}")

    ############################################
    ### Utils
    ############################################

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

    def _read_from_cache(
        self,
        alias: str,
        version: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Optional[CachedPrompt]:
        if portalocker is None or not os.path.exists(CACHE_FILE_NAME):
            return None

        try:
            # Use shared lock for reading to allow concurrent reads
            with portalocker.Lock(
                CACHE_FILE_NAME,
                mode="r",
                flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
            ) as f:
                cache_data = json.load(f)

            if alias in cache_data:
                if version:
                    if (
                        VERSION_CACHE_KEY in cache_data[alias]
                        and version in cache_data[alias][VERSION_CACHE_KEY]
                    ):
                        return CachedPrompt(
                            **cache_data[alias][VERSION_CACHE_KEY][version]
                        )
                elif label:
                    if (
                        LABEL_CACHE_KEY in cache_data[alias]
                        and label in cache_data[alias][LABEL_CACHE_KEY]
                    ):
                        return CachedPrompt(
                            **cache_data[alias][LABEL_CACHE_KEY][label]
                        )
            return None
        except (portalocker.exceptions.LockException, Exception):
            # If cache is locked, corrupted or unreadable, return None and let it fetch from API
            return None

    def _write_to_cache(
        self,
        cache_key: Literal[VERSION_CACHE_KEY, LABEL_CACHE_KEY],
        version: str,
        label: Optional[str] = None,
        text_template: Optional[str] = None,
        messages_template: Optional[List[PromptMessage]] = None,
        prompt_version_id: Optional[str] = None,
        type: Optional[PromptType] = None,
        interpolation_type: Optional[PromptInterpolationType] = None,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[OutputType] = None,
        output_schema: Optional[OutputSchema] = None,
    ):
        if portalocker is None or not self.alias:
            return

        try:
            # Ensure directory exists
            os.makedirs(HIDDEN_DIR, exist_ok=True)
            # Use r+ mode if file exists, w mode if it doesn't
            mode = "r+" if os.path.exists(CACHE_FILE_NAME) else "w"

            with portalocker.Lock(
                CACHE_FILE_NAME,
                mode=mode,
                flags=portalocker.LOCK_EX,
            ) as f:
                # Read existing cache data if file exists and has content
                cache_data = {}
                if mode == "r+":
                    try:
                        f.seek(0)
                        content = f.read()
                        if content:
                            cache_data = json.loads(content)
                    except (json.JSONDecodeError, Exception):
                        cache_data = {}

                # Ensure the cache structure is initialized properly
                if self.alias not in cache_data:
                    cache_data[self.alias] = {}

                if cache_key not in cache_data[self.alias]:
                    cache_data[self.alias][cache_key] = {}

                # Cache the prompt
                cached_entry = {
                    "alias": self.alias,
                    "version": version,
                    "label": label,
                    "template": text_template,
                    "messages_template": messages_template,
                    "prompt_version_id": prompt_version_id,
                    "type": type,
                    "interpolation_type": interpolation_type,
                    "model_settings": model_settings,
                    "output_type": output_type,
                    "output_schema": output_schema,
                }

                if cache_key == VERSION_CACHE_KEY:
                    cache_data[self.alias][cache_key][version] = cached_entry
                else:
                    cache_data[self.alias][cache_key][label] = cached_entry

                # Write back to cache file
                f.seek(0)
                f.truncate()
                json.dump(cache_data, f, cls=CustomEncoder)
        except portalocker.exceptions.LockException:
            # If we can't acquire the lock, silently skip caching
            pass
        except Exception:
            # If any other error occurs during caching, silently skip
            pass

    def _load_from_cache_with_progress(
        self,
        progress: Progress,
        task_id: int,
        start_time: float,
        version: Optional[str] = None,
        label: Optional[str] = None,
    ):
        """
        Load prompt from cache and update progress bar.
        Raises if unable to load from cache.
        """
        cached_prompt = self._read_from_cache(
            self.alias, version=version, label=label
        )
        if not cached_prompt:
            raise ValueError("Unable to fetch prompt and load from cache")

        with self._lock:
            self._version = cached_prompt.version
            self.label = cached_prompt.label
            self.text_template = cached_prompt.template
            self.messages_template = cached_prompt.messages_template
            self._prompt_version_id = cached_prompt.prompt_version_id
            self.type = (
                PromptType(cached_prompt.type) if cached_prompt.type else None
            )
            self.interpolation_type = (
                PromptInterpolationType(cached_prompt.interpolation_type)
                if cached_prompt.interpolation_type
                else None
            )
            self.model_settings = cached_prompt.model_settings
            self.output_type = (
                OutputType(cached_prompt.output_type)
                if cached_prompt.output_type
                else None
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

    ############################################
    ### Pull, Push, Update
    ############################################

    def pull(
        self,
        version: Optional[str] = None,
        label: Optional[str] = None,
        fallback_to_cache: bool = True,
        write_to_cache: bool = True,
        default_to_cache: bool = True,
        refresh: Optional[int] = 60,
    ):
        should_write_on_first_fetch = False
        if refresh:
            # Check if we need to bootstrap the cache
            cached_prompt = self._read_from_cache(
                self.alias, version=version, label=label
            )
            if cached_prompt is None:
                # No cache exists, so we should write after fetching to bootstrap
                should_write_on_first_fetch = True
            write_to_cache = False  # Polling will handle subsequent writes

        if self.alias is None:
            raise TypeError(
                "Unable to pull prompt from Confident AI when no alias is provided."
            )

        # Manage background prompt polling
        if refresh:
            loop = _get_or_create_polling_loop()
            asyncio.run_coroutine_threadsafe(
                self.create_polling_task(version, label, refresh), loop
            )

        if default_to_cache:
            try:
                cached_prompt = self._read_from_cache(
                    self.alias, version=version, label=label
                )
                if cached_prompt:
                    with self._lock:
                        self._version = cached_prompt.version
                        self.label = cached_prompt.label
                        self.text_template = cached_prompt.template
                        self.messages_template = cached_prompt.messages_template
                        self._prompt_version_id = (
                            cached_prompt.prompt_version_id
                        )
                        self.type = (
                            PromptType(cached_prompt.type)
                            if cached_prompt.type
                            else None
                        )
                        self.interpolation_type = (
                            PromptInterpolationType(
                                cached_prompt.interpolation_type
                            )
                            if cached_prompt.interpolation_type
                            else None
                        )
                        self.model_settings = cached_prompt.model_settings
                        self.output_type = (
                            OutputType(cached_prompt.output_type)
                            if cached_prompt.output_type
                            else None
                        )
                        self.output_schema = construct_base_model(
                            cached_prompt.output_schema
                        )
                    return
            except Exception:
                pass

        api = Api()
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            BarColumn(bar_width=60),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            HINT_TEXT = (
                f"version='{version or 'latest'}'"
                if not label
                else f"label='{label}'"
            )
            task_id = progress.add_task(
                f"Pulling [rgb(106,0,255)]'{self.alias}' ({HINT_TEXT})[/rgb(106,0,255)] from Confident AI...",
                total=100,
            )

            start_time = time.perf_counter()
            try:
                if label:
                    data, _ = api.send_request(
                        method=HttpMethods.GET,
                        endpoint=Endpoints.PROMPTS_LABEL_ENDPOINT,
                        url_params={
                            "alias": self.alias,
                            "label": label,
                        },
                    )
                else:
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
                    version=data.get("version", None),
                    label=data.get("label", None),
                    text=data.get("text", None),
                    messages=data.get("messages", None),
                    type=data["type"],
                    interpolation_type=data["interpolationType"],
                    model_settings=data.get("modelSettings", None),
                    output_type=data.get("outputType", None),
                    output_schema=data.get("outputSchema", None),
                )
            except Exception:
                if fallback_to_cache:
                    self._load_from_cache_with_progress(
                        progress,
                        task_id,
                        start_time,
                        version=version,
                        label=label,
                    )
                    return
                raise

            with self._lock:
                self._version = response.version
                self.label = response.label
                self.text_template = response.text
                self.messages_template = response.messages
                self._prompt_version_id = response.id
                self.type = response.type
                self.interpolation_type = response.interpolation_type
                self.model_settings = response.model_settings
                self.output_type = response.output_type
                self.output_schema = construct_base_model(
                    response.output_schema
                )

            end_time = time.perf_counter()
            time_taken = format(end_time - start_time, ".2f")
            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Done! ({time_taken}s)",
            )
            # Write to cache if explicitly requested OR if we need to bootstrap cache for refresh mode
            if write_to_cache or should_write_on_first_fetch:
                self._write_to_cache(
                    cache_key=LABEL_CACHE_KEY if label else VERSION_CACHE_KEY,
                    version=response.version,
                    label=response.label,
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
        _verbose: Optional[bool] = True,
    ):
        if self.alias is None:
            raise ValueError(
                "Prompt alias is not set. Please set an alias to continue."
            )
        text_template = text or self.text_template
        messages_template = messages or self.messages_template
        if text_template is None and messages_template is None:
            raise ValueError("Either text or messages must be provided")
        if text_template is not None and messages_template is not None:
            raise ValueError("Only one of text or messages can be provided")

        body = PromptPushRequest(
            alias=self.alias,
            text=text_template,
            messages=messages_template,
            interpolation_type=interpolation_type or self.interpolation_type,
            model_settings=model_settings or self.model_settings,
            output_type=output_type or self.output_type,
            output_schema=construct_output_schema(output_schema)
            or construct_output_schema(self.output_schema),
        )
        try:
            body = body.model_dump(
                by_alias=True, exclude_none=True, mode="json"
            )
        except AttributeError:
            # Pydantic version below 2.0
            body = body.dict(by_alias=True, exclude_none=True)

        api = Api()
        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.PROMPTS_ENDPOINT,
            body=body,
        )
        versions = self._get_versions()

        if link and versions:
            self._prompt_version_id = versions[-1].id
            self.text_template = text_template
            self.messages_template = messages_template
            self.interpolation_type = (
                interpolation_type or self.interpolation_type
            )
            self.model_settings = model_settings or self.model_settings
            self.output_type = output_type or self.output_type
            self.output_schema = output_schema or self.output_schema
            self.type = PromptType.TEXT if text_template else PromptType.LIST
            if _verbose:
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
            body = body.model_dump(
                by_alias=True, exclude_none=True, mode="json"
            )
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
            self._version = version
            self.text_template = text
            self.messages_template = messages
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
        label: Optional[str],
        refresh: Optional[int] = 60,
    ):
        # If polling task doesn't exist, start it
        CACHE_KEY = LABEL_CACHE_KEY if label else VERSION_CACHE_KEY
        cache_value = label if label else version

        # Initialize nested dicts if they don't exist
        if CACHE_KEY not in self._polling_tasks:
            self._polling_tasks[CACHE_KEY] = {}
        if CACHE_KEY not in self._refresh_map:
            self._refresh_map[CACHE_KEY] = {}

        polling_task: Optional[asyncio.Task] = self._polling_tasks[
            CACHE_KEY
        ].get(cache_value)

        if refresh:
            self._refresh_map[CACHE_KEY][cache_value] = refresh
            if not polling_task:
                self._polling_tasks[CACHE_KEY][cache_value] = (
                    asyncio.create_task(self.poll(version, label))
                )

        # If invalid `refresh`, stop the task
        else:
            if polling_task:
                polling_task.cancel()
            if cache_value in self._polling_tasks[CACHE_KEY]:
                self._polling_tasks[CACHE_KEY].pop(cache_value)
            if cache_value in self._refresh_map[CACHE_KEY]:
                self._refresh_map[CACHE_KEY].pop(cache_value)

    async def poll(
        self,
        version: Optional[str] = None,
        label: Optional[str] = None,
    ):
        CACHE_KEY = LABEL_CACHE_KEY if label else VERSION_CACHE_KEY
        cache_value = label if label else version

        while True:
            await asyncio.sleep(self._refresh_map[CACHE_KEY][cache_value])

            api = Api()
            try:
                if label:
                    data, _ = api.send_request(
                        method=HttpMethods.GET,
                        endpoint=Endpoints.PROMPTS_LABEL_ENDPOINT,
                        url_params={
                            "alias": self.alias,
                            "label": label,
                        },
                    )
                else:
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
                    version=data.get("version", None),
                    label=data.get("label", None),
                    text=data.get("text", None),
                    messages=data.get("messages", None),
                    type=data["type"],
                    interpolation_type=data["interpolationType"],
                )

                # Update the cache with fresh data from server
                self._write_to_cache(
                    cache_key=CACHE_KEY,
                    version=response.version,
                    label=response.label,
                    text_template=response.text,
                    messages_template=response.messages,
                    prompt_version_id=response.id,
                    type=response.type,
                    interpolation_type=response.interpolation_type,
                )

                # Update in-memory properties with fresh data (thread-safe)
                with self._lock:
                    self._version = response.version
                    self.label = response.label
                    self.text_template = response.text
                    self.messages_template = response.messages
                    self._prompt_version_id = response.id
                    self.type = response.type
                    self.interpolation_type = response.interpolation_type

            except Exception:
                pass

    def _stop_polling(self):
        loop = _polling_loop
        if not loop or not loop.is_running():
            return

        # Stop all polling tasks
        for ck in list(self._polling_tasks.keys()):
            for cv in list(self._polling_tasks[ck].keys()):
                task = self._polling_tasks[ck][cv]
                if task and not task.done():
                    loop.call_soon_threadsafe(task.cancel)
            self._polling_tasks[ck].clear()
            self._refresh_map[ck].clear()
        return
