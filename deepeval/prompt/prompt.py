from enum import Enum
from typing import Literal, Optional, List, Dict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
import time
import json
import os
from pydantic import BaseModel
import asyncio
import portalocker
import threading

from deepeval.prompt.api import (
    PromptHttpResponse,
    PromptMessage,
    PromptType,
    PromptInterpolationType,
    PromptPushRequest,
    PromptVersionsHttpResponse,
)
from deepeval.prompt.utils import interpolate_text
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.constants import HIDDEN_DIR

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
    alias: str
    version: str
    label: Optional[str] = None
    template: Optional[str]
    messages_template: Optional[List[PromptMessage]]
    prompt_version_id: str
    type: PromptType
    interpolation_type: PromptInterpolationType

    class Config:
        use_enum_values = True


class Prompt:
    label: Optional[str] = None
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
        if template and messages_template:
            raise TypeError(
                "Unable to create Prompt where 'template' and 'messages_template' are both provided. Please provide only one to continue."
            )

        self.alias = alias
        self._text_template = template
        self._messages_template = messages_template
        self._version = None
        self._polling_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        self._refresh_map: Dict[str, Dict[str, int]] = {}
        self._lock = (
            threading.Lock()
        )  # Protect instance attributes from race conditions
        if template:
            self._type = PromptType.TEXT
        elif messages_template:
            self._type = PromptType.LIST

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

    def interpolate(self, **kwargs):
        with self._lock:
            prompt_type = self._type
            text_template = self._text_template
            messages_template = self._messages_template
            interpolation_type = self._interpolation_type

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
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

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
        if not os.path.exists(CACHE_FILE_NAME):
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
    ):
        if not self.alias:
            return

        # Ensure directory exists
        os.makedirs(HIDDEN_DIR, exist_ok=True)

        try:
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
            self.version = cached_prompt.version
            self.label = cached_prompt.label
            self._text_template = cached_prompt.template
            self._messages_template = cached_prompt.messages_template
            self._prompt_version_id = cached_prompt.prompt_version_id
            self._type = PromptType(cached_prompt.type)
            self._interpolation_type = PromptInterpolationType(
                cached_prompt.interpolation_type
            )

        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description}[rgb(25,227,160)]Loaded from cache! ({time_taken}s)",
        )

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
                        self.version = cached_prompt.version
                        self.label = cached_prompt.label
                        self._text_template = cached_prompt.template
                        self._messages_template = (
                            cached_prompt.messages_template
                        )
                        self._prompt_version_id = (
                            cached_prompt.prompt_version_id
                        )
                        self._type = PromptType(cached_prompt.type)
                        self._interpolation_type = PromptInterpolationType(
                            cached_prompt.interpolation_type
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
                self.version = response.version
                self.label = response.label
                self._text_template = response.text
                self._messages_template = response.messages
                self._prompt_version_id = response.id
                self._type = response.type
                self._interpolation_type = response.interpolation_type

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
                )

    def push(
        self,
        text: Optional[str] = None,
        messages: Optional[List[PromptMessage]] = None,
        interpolation_type: Optional[
            PromptInterpolationType
        ] = PromptInterpolationType.FSTRING,
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
            console = Console()
            console.print(
                "✅ Prompt successfully pushed to Confident AI! View at "
                f"[link={link}]{link}[/link]"
            )

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
                    self.version = response.version
                    self.label = response.label
                    self._text_template = response.text
                    self._messages_template = response.messages
                    self._prompt_version_id = response.id
                    self._type = response.type
                    self._interpolation_type = response.interpolation_type

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
