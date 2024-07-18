from contextvars import ContextVar
from enum import Enum
import copy
import os
import json
import time
from typing import Any, Optional, Dict, List, Union
from collections.abc import Iterable
import tqdm
import re
import string
import numpy as np
from dataclasses import asdict, is_dataclass
import re
import asyncio
import nest_asyncio
import uuid
from pydantic import BaseModel

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def convert_keys_to_snake_case(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            camel_to_snake(k): convert_keys_to_snake_case(v)
            for k, v in data.items()
        }
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
            formatted_elements.append(
                json.dumps(item.dict(), indent=4).replace("\n", "\n    ")
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
            print(
                "Event loop is already running. Applying nest_asyncio patch to allow async execution..."
            )
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def show_indicator():
    try:
        if os.environ["DISABLE_DEEPEVAL_INDICATOR"] == "YES":
            return False
        else:
            return True
    except:
        return True


def set_indicator(show_indicator: bool):
    if show_indicator:
        os.environ["DISABLE_DEEPEVAL_INDICATOR"] = "NO"
    else:
        os.environ["DISABLE_DEEPEVAL_INDICATOR"] = "YES"


def should_ignore_errors():
    try:
        if os.environ["IGNORE_DEEPEVAL_ERRORS"] == "YES":
            return True
        else:
            return False
    except:
        return False


def set_should_ignore_errors(yes: bool):
    if yes:
        os.environ["IGNORE_DEEPEVAL_ERRORS"] = "YES"
    else:
        os.environ["IGNORE_DEEPEVAL_ERRORS"] = "NO"


def should_verbose_print() -> Union[bool, None]:
    try:
        if os.environ["DEEPEVAL_VERBOSE_MODE"] == "YES":
            return True
        else:
            return None
    except:
        return None


def set_verbose_mode(yes: Optional[bool]):
    if yes:
        os.environ["DEEPEVAL_VERBOSE_MODE"] = "YES"


def should_use_cache():
    try:
        if os.environ["ENABLE_DEEPEVAL_CACHE"] == "YES":
            return True
        else:
            return False
    except:
        return False


def set_should_use_cache(yes: bool):
    if yes:
        os.environ["ENABLE_DEEPEVAL_CACHE"] = "YES"
    else:
        os.environ["ENABLE_DEEPEVAL_CACHE"] = "NO"


def login_with_confident_api_key(api_key: string):
    from rich import print

    KEY_FILE_HANDLER.write_key(KeyValues.API_KEY, api_key)
    print("Congratulations! Login successful :raising_hands: ")


def set_is_running_deepeval(flag: bool):
    if flag:
        os.environ["DEEPEVAL"] = "YES"
    else:
        os.environ["DEEPEVAL"] = "NO"


def get_is_running_deepeval() -> bool:
    try:
        return os.environ["DEEPEVAL"] == "YES"
    except:
        return False


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


def is_confident():
    confident_api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY)
    return confident_api_key is not None


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
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def cosine_similarity(vector_a, vector_b):
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


###############################################
# Source: https://github.com/tingofurro/summac
###############################################

# GPU-related business


def get_freer_gpu():
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" + freer_gpu
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
