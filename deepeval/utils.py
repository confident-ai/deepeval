import copy
import os
import json
import time
import webbrowser
import tqdm
import re
import string
import asyncio
import nest_asyncio
import uuid
import math

from contextvars import ContextVar
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel
from rich.progress import Progress
from rich.console import Console, Theme

from deepeval.confident.api import set_confident_api_key
from deepeval.config.settings import get_settings
from deepeval.config.utils import (
    get_env_bool,
    set_env_bool,
)


###############
# Local Types #
###############


class TurnLike(Protocol):
    order: int
    role: str
    content: str
    user_id: Optional[str]
    retrieval_context: Optional[Sequence[str]]
    tools_called: Optional[Sequence[Any]]
    additional_metadata: Optional[Dict[str, Any]]
    comments: Optional[str]


def get_lcs(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def convert_keys_to_snake_case(data: Any) -> Any:
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = camel_to_snake(k)
            if k == "additionalMetadata":
                new_dict[new_key] = (
                    v  # Convert key but do not recurse into value
                )
            else:
                new_dict[new_key] = convert_keys_to_snake_case(v)
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_snake_case(i) for i in data]
    else:
        return data


def prettify_list(lst: List[Any]):
    if len(lst) == 0:
        return "[]"

    formatted_elements = []
    for item in lst:
        if isinstance(item, str):
            formatted_elements.append(f'"{item}"')
        elif isinstance(item, BaseModel):
            try:
                jsonObj = item.model_dump()
            except AttributeError:
                # Pydantic version below 2.0
                jsonObj = item.dict()

            formatted_elements.append(
                json.dumps(jsonObj, indent=4, ensure_ascii=True).replace(
                    "\n", "\n    "
                )
            )
        else:
            formatted_elements.append(repr(item))  # Fallback for other types

    formatted_list = ",\n    ".join(formatted_elements)
    return f"[\n    {formatted_list}\n]"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_dict_with_sorting(obj):
    if obj is None:
        return obj
    elif isinstance(obj, dict):
        sorted_dict = {
            k: serialize_dict_with_sorting(v) for k, v in sorted(obj.items())
        }
        return sorted_dict
    elif isinstance(obj, list):
        sorted_list = sorted(
            [serialize_dict_with_sorting(item) for item in obj],
            key=lambda x: json.dumps(x),
        )
        return sorted_list
    else:
        return obj


def serialize(obj) -> Union[str, None]:
    return json.dumps(serialize_dict_with_sorting(obj), sort_keys=True)


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_or_create_general_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def set_should_skip_on_missing_params(yes: bool):
    s = get_settings()
    with s.edit(persist=False):
        s.SKIP_DEEPEVAL_MISSING_PARAMS = yes


def should_ignore_errors() -> bool:
    return bool(get_settings().IGNORE_DEEPEVAL_ERRORS)


def should_skip_on_missing_params() -> bool:
    return bool(get_settings().SKIP_DEEPEVAL_MISSING_PARAMS)


def set_should_ignore_errors(yes: bool):
    s = get_settings()
    with s.edit(persist=False):
        s.IGNORE_DEEPEVAL_ERRORS = yes


def should_verbose_print() -> bool:
    return bool(get_settings().DEEPEVAL_VERBOSE_MODE)


def set_verbose_mode(yes: Optional[bool]):
    s = get_settings()
    with s.edit(persist=False):
        s.DEEPEVAL_VERBOSE_MODE = yes


def set_identifier(identifier: Optional[str]):
    if identifier:
        s = get_settings()
        with s.edit(persist=False):
            s.DEEPEVAL_IDENTIFIER = identifier


def get_identifier() -> Optional[str]:
    return get_settings().DEEPEVAL_IDENTIFIER


def should_use_cache() -> bool:
    return bool(get_settings().ENABLE_DEEPEVAL_CACHE)


def set_should_use_cache(yes: bool):
    s = get_settings()
    with s.edit(persist=False):
        s.ENABLE_DEEPEVAL_CACHE = yes


def login(api_key: str):
    if not api_key or not isinstance(api_key, str):
        raise ValueError("Oh no! Please provide an api key string to login.")
    elif len(api_key) == 0:
        raise ValueError("Unable to login, please provide a non-empty api key.")

    from rich import print

    set_confident_api_key(api_key)
    print(
        "🎉🥳 Congratulations! You've successfully logged in! :raising_hands: "
    )


def set_is_running_deepeval(flag: bool):
    set_env_bool("DEEPEVAL", flag)


def get_is_running_deepeval() -> bool:
    return get_env_bool("DEEPEVAL")


def is_in_ci_env() -> bool:
    ci_env_vars = [
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "CIRCLECI",  # CircleCI
        "JENKINS_URL",  # Jenkins
        "TRAVIS",  # Travis CI
        "CI",  # Generic CI indicator used by many services
        "CONTINUOUS_INTEGRATION",  # Another generic CI indicator
        "TEAMCITY_VERSION",  # TeamCity
        "BUILDKITE",  # Buildkite
        "BITBUCKET_BUILD_NUMBER",  # Bitbucket Pipelines
        "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",  # Azure Pipelines
        "HEROKU_TEST_RUN_ID",  # Heroku CI
    ]

    for var in ci_env_vars:
        if os.getenv(var) is not None:
            return True

    return False


def open_browser(url: str):
    if get_settings().CONFIDENT_OPEN_BROWSER:
        if not is_in_ci_env():
            webbrowser.open(url)


def capture_contextvars(single_obj):
    contextvars_dict = {}
    for attr in dir(single_obj):
        attr_value = getattr(single_obj, attr, None)
        if isinstance(attr_value, ContextVar):
            contextvars_dict[attr] = (attr_value, attr_value.get())
    return contextvars_dict


def update_contextvars(single_obj, contextvars_dict):
    for attr, (context_var, value) in contextvars_dict.items():
        context_var.set(value)
        setattr(single_obj, attr, context_var)


def drop_and_copy(obj, drop_attrs):
    # Function to drop attributes from a single object
    def drop_attrs_from_single_obj(single_obj, drop_attrs):
        temp_attrs = {}
        for attr in drop_attrs:
            if hasattr(single_obj, attr):
                temp_attrs[attr] = getattr(single_obj, attr)
                delattr(single_obj, attr)
        return temp_attrs

    # Function to remove ContextVar attributes from a single object
    def remove_contextvars(single_obj):
        temp_contextvars = {}
        for attr in dir(single_obj):
            if isinstance(getattr(single_obj, attr, None), ContextVar):
                temp_contextvars[attr] = getattr(single_obj, attr)
                delattr(single_obj, attr)
        return temp_contextvars

    # Function to restore ContextVar attributes to a single object
    def restore_contextvars(single_obj, contextvars):
        for attr, value in contextvars.items():
            setattr(single_obj, attr, value)

    # Check if obj is iterable (but not a string)
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        copied_objs = []
        for item in obj:
            temp_attrs = drop_attrs_from_single_obj(item, drop_attrs)
            temp_contextvars = remove_contextvars(item)
            copied_obj = copy.deepcopy(item)
            restore_contextvars(copied_obj, temp_contextvars)

            # Restore attributes to the original object
            for attr, value in temp_attrs.items():
                setattr(item, attr, value)
            restore_contextvars(item, temp_contextvars)

            copied_objs.append(copied_obj)

        return copied_objs
    else:
        temp_attrs = drop_attrs_from_single_obj(obj, drop_attrs)
        temp_contextvars = remove_contextvars(obj)
        copied_obj = copy.deepcopy(obj)
        restore_contextvars(copied_obj, temp_contextvars)

        # Restore attributes to the original object
        for attr, value in temp_attrs.items():
            setattr(obj, attr, value)
        restore_contextvars(obj, temp_contextvars)

        return copied_obj


def dataclass_to_dict(instance: Any) -> Any:
    if is_dataclass(instance):
        return {k: dataclass_to_dict(v) for k, v in asdict(instance).items()}
    elif isinstance(instance, Enum):
        return instance.value
    elif isinstance(instance, list):
        return [dataclass_to_dict(item) for item in instance]
    elif isinstance(instance, tuple):
        return tuple(dataclass_to_dict(item) for item in instance)
    elif isinstance(instance, dict):
        return {k: dataclass_to_dict(v) for k, v in instance.items()}
    else:
        return instance


def class_to_dict(instance: Any) -> Any:
    if isinstance(instance, Enum):
        return instance.value
    elif isinstance(instance, list):
        return [class_to_dict(item) for item in instance]
    elif isinstance(instance, tuple):
        return tuple(class_to_dict(item) for item in instance)
    elif isinstance(instance, dict):
        return {k: class_to_dict(v) for k, v in instance.items()}
    elif hasattr(instance, "__dict__"):
        instance_dict: Dict = instance.__dict__
        return {str(k): class_to_dict(v) for k, v in instance_dict.items()}
    else:
        return instance


def delete_file_if_exists(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def softmax(x):
    import numpy as np

    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def cosine_similarity(vector_a, vector_b):
    import numpy as np

    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def chunk_text(text, chunk_size=20):
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def is_missing(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")


def len_tiny() -> int:
    value = get_settings().DEEPEVAL_MAXLEN_TINY
    return value if (isinstance(value, int) and value > 0) else 40


def len_short() -> int:
    value = get_settings().DEEPEVAL_MAXLEN_SHORT
    return value if (isinstance(value, int) and value > 0) else 60


def len_medium() -> int:
    value = get_settings().DEEPEVAL_MAXLEN_MEDIUM
    return value if (isinstance(value, int) and value > 0) else 120


def len_long() -> int:
    value = get_settings().DEEPEVAL_MAXLEN_LONG
    return value if (isinstance(value, int) and value > 0) else 240


def shorten(
    text: Optional[object],
    max_len: Optional[int] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Truncate text to max_len characters, appending `suffix` if truncated.
    - Accepts None and returns "", or any object is returned as str().
    - Safe when max_len <= len(suffix).
    """
    settings = get_settings()

    if max_len is None:
        max_len = (
            settings.DEEPEVAL_SHORTEN_DEFAULT_MAXLEN
            if settings.DEEPEVAL_SHORTEN_DEFAULT_MAXLEN is not None
            else len_long()
        )
    if suffix is None:
        suffix = (
            settings.DEEPEVAL_SHORTEN_SUFFIX
            if settings.DEEPEVAL_SHORTEN_SUFFIX is not None
            else "..."
        )

    if text is None:
        return ""
    stext = str(text)
    if max_len <= 0:
        return ""
    if len(stext) <= max_len:
        return stext
    cut = max_len - len(suffix)
    if cut <= 0:
        return suffix[:max_len]
    return stext[:cut] + suffix


def format_turn(
    turn: TurnLike,
    *,
    content_length: Optional[int] = None,
    max_context_items: Optional[int] = None,
    context_length: Optional[int] = None,
    meta_length: Optional[int] = None,
    include_tools_in_header: bool = True,
    include_order_role_in_header: bool = True,
) -> str:
    """
    Build a multi-line, human-readable summary for a conversational turn.
    Safe against missing fields and overly long content.
    """
    if content_length is None:
        content_length = len_long()
    if max_context_items is None:
        max_context_items = 2
    if context_length is None:
        context_length = len_medium()
    if meta_length is None:
        meta_length = len_medium()

    tools = turn.tools_called or []
    tool_names = ", ".join(getattr(tc, "name", str(tc)) for tc in tools)
    content = shorten(turn.content, content_length)

    lines = []

    if include_order_role_in_header:
        header = f"{turn.order:>2}. {turn.role:<9} {content}"
        if include_tools_in_header and tool_names:
            header += f"  | tools: {tool_names}"
        if turn.user_id:
            header += f"  | user: {shorten(turn.user_id, len_tiny())}"
        lines.append(header)
        indent = "      "
    else:
        # No order or role prefix in this mode
        # keep tools out of header as well.
        first = content
        if turn.user_id:
            first += f"  | user: {shorten(turn.user_id, len_tiny())}"
        lines.append(first)
        indent = "      "  # ctx and meta indent

    rctx = list(turn.retrieval_context or [])
    if rctx:
        show = rctx[:max_context_items]
        for i, item in enumerate(show):
            lines.append(f"{indent}↳ ctx[{i}]: {shorten(item, context_length)}")
        hidden = max(0, len(rctx) - len(show))
        if hidden:
            lines.append(f"{indent}↳ ctx: (+{hidden} more)")

    if turn.comments:
        lines.append(
            f"{indent}↳ comment: {shorten(str(turn.comments), meta_length)}"
        )

    meta = turn.additional_metadata or {}
    if isinstance(meta, dict):
        for k in list(meta.keys())[:3]:
            if k in {"user_id", "userId"}:
                continue
            v = meta.get(k)
            if v is not None:
                lines.append(
                    f"{indent}↳ meta.{k}: {shorten(str(v), meta_length)}"
                )

    return "\n".join(lines)


###############################################
# Source: https://github.com/tingofurro/summac
###############################################

# GPU-related business


def get_freer_gpu():
    import numpy as np

    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [
        int(x.split()[2]) + 5 * i
        for i, x in enumerate(open("tmp_smi", "r").readlines())
    ]
    os.remove("tmp_smi")
    return np.argmax(memory_available)


def any_gpu_with_space(gb_needed):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [
        float(x.split()[2]) / 1024.0
        for i, x in enumerate(open("tmp_smi", "r").readlines())
    ]
    os.remove("tmp_smi")
    return any([mem >= gb_needed for mem in memory_available])


def wait_free_gpu(gb_needed):
    while not any_gpu_with_space(gb_needed):
        time.sleep(30)


def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    print("Will use GPU: %s" % (freer_gpu))

    s = get_settings()
    with s.edit(persist=False):
        s.CUDA_LAUNCH_BLOCKING = True
        s.CUDA_VISIBLE_DEVICES = freer_gpu
    return freer_gpu


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch


def clean_nested_dict(data):
    if isinstance(data, dict):
        return {key: clean_nested_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_nested_dict(item) for item in data]
    elif isinstance(data, str):
        return data.replace("\x00", "")
    else:
        return data


def update_pbar(
    progress: Optional[Progress],
    pbar_id: Optional[int],
    advance: int = 1,
    advance_to_end: bool = False,
    remove: bool = True,
    total: Optional[int] = None,
):
    if progress is None or pbar_id is None:
        return
    # Get amount to advance
    current_task = next(t for t in progress.tasks if t.id == pbar_id)
    if advance_to_end:
        advance = current_task.remaining
    # Advance
    progress.update(pbar_id, advance=advance, total=total)
    # Remove if finished
    if current_task.finished and remove:
        progress.remove_task(pbar_id)


def add_pbar(progress: Optional[Progress], description: str, total: int = 1):
    if progress is None:
        return None
    return progress.add_task(description, total=total)


def remove_pbars(
    progress: Optional[Progress], pbar_ids: List[int], cascade: bool = True
):
    if progress is None:
        return
    for pbar_id in pbar_ids:
        if cascade:
            time.sleep(0.1)
        progress.remove_task(pbar_id)


def read_env_int(
    name: str, default: int, *, min_value: Union[int, None] = None
) -> int:
    """Read an integer from an environment variable with safe fallback.

    Attempts to read os.environ[name] and parse it as an int. If the variable
    is unset, cannot be parsed, or is less than `min_value` (when provided),
    the function returns `default`.

    Args:
        name: Environment variable name to read.
        default: Value to return when the env var is missing/invalid/out of range.
        min_value: Optional inclusive lower bound; values < min_value are rejected.

    Returns:
        The parsed integer, or `default` on any failure.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw)
        if min_value is not None and v < min_value:
            return default
        return v
    except Exception:
        return default


def read_env_float(
    name: str, default: float, *, min_value: Union[float, None] = None
) -> float:
    """Read a float from an environment variable with safe fallback.

    Attempts to read os.environ[name] and parse it as a float. If the variable
    is unset, cannot be parsed, or is less than `min_value` (when provided),
    the function returns `default`.

    Args:
        name: Environment variable name to read.
        default: Value to return when the env var is missing/invalid/out of range.
        min_value: Optional inclusive lower bound; values < min_value are rejected.

    Returns:
        The parsed float, or `default` on any failure.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except Exception:
        return default

    if not math.isfinite(v):
        return default
    if min_value is not None and v < min_value:
        return default
    return v


my_theme = Theme(
    {
        "bar.complete": "#11ff00",
        "progress.percentage": "#00e5ff",
        # "progress.data.speed": "#00FF00",
        # "progress.remaining": "#00FF00",
        "progress.elapsed": "#5703ff",
    }
)
custom_console = Console(theme=my_theme)
