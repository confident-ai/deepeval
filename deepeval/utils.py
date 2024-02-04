from enum import Enum
import copy
import os
import time
from typing import Any, Optional, Dict
from collections.abc import Iterable
import tqdm
import re
import string
import numpy as np
from dataclasses import asdict, is_dataclass
import re

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


def get_deployment_configs() -> Optional[Dict]:
    if os.getenv("GITHUB_ACTIONS") == "true":
        env_info = {
            "env": "GitHub Actions",
            "actor": os.getenv("GITHUB_ACTOR", None),
            "sha": os.getenv("GITHUB_SHA", None),
            "repo": os.getenv("GITHUB_REPOSITORY", None),
        }

        branch_ref = os.getenv("GITHUB_REF", "")
        if branch_ref.startswith("refs/pull/"):
            is_pull_request = True
        else:
            is_pull_request = False

        env_info["is_pull_request"] = is_pull_request
        env_info["branch"] = (
            branch_ref.replace("refs/heads/", "") if branch_ref else None
        )
        return env_info

    return None


def is_confident():
    confident_api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY)
    return confident_api_key is not None


def drop_and_copy(obj, drop_attrs):
    # Function to drop attributes from a single object
    def drop_attrs_from_single_obj(single_obj, drop_attrs):
        temp_attrs = {}
        for attr in drop_attrs:
            if hasattr(single_obj, attr):
                temp_attrs[attr] = getattr(single_obj, attr)
                delattr(single_obj, attr)
        return temp_attrs

    # Check if obj is iterable (but not a string)
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        copied_objs = []
        for item in obj:
            temp_attrs = drop_attrs_from_single_obj(item, drop_attrs)
            copied_objs.append(copy.deepcopy(item))
            # Restore attributes to the original item
            for attr, value in temp_attrs.items():
                setattr(item, attr, value)
        return copied_objs
    else:
        # If obj is not iterable, apply directly
        temp_attrs = drop_attrs_from_single_obj(obj, drop_attrs)
        copied_obj = copy.deepcopy(obj)
        # Restore attributes to the original object
        for attr, value in temp_attrs.items():
            setattr(obj, attr, value)
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


def trimToJson(input_string: str) -> str:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1
    return input_string[start:end] if start != -1 and end != 0 else ""


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
