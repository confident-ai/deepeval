from collections.abc import Callable
from functools import wraps
from typing import Dict, List, Optional, Union

from deepeval.confident.api import is_confident
from deepeval.prompt import Prompt
from deepeval.prompt.api import PromptApi
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.test_run import TEMP_FILE_PATH, PromptData


def process_hyperparameters(
    hyperparameters: Optional[Dict] = None,
    verbose: bool = True,
) -> Union[Dict[str, Union[str, int, float, PromptApi]], None]:
    if hyperparameters is None:
        return None

    if not isinstance(hyperparameters, dict):
        raise TypeError("Hyperparameters must be a dictionary or None")

    processed_hyperparameters = {}
    prompts_hash_id_map = {}

    for key, value in hyperparameters.items():
        if not isinstance(key, str):
            raise TypeError(f"Hyperparameter key '{key}' must be a string")

        if value is None:
            continue

        if not isinstance(value, (str, int, float, Prompt)):
            raise TypeError(
                f"Hyperparameter value for key '{key}' must be a string, integer, float, or Prompt"
            )

        if isinstance(value, Prompt):
            try:
                prompt_key = f"{value.alias}_{value.hash}"
            except Exception:
                prompt_key = f"{value.alias}_[hash]"

            if value._prompt_id is not None and value.type is not None:
                processed_hyperparameters[key] = PromptApi(
                    id=value.hash,
                    type=value.type,
                )
            elif is_confident():
                if prompt_key not in prompts_hash_id_map:
                    value.push(_verbose=verbose)
                    prompt_key = prompt_key.replace("[hash]", value.hash)
                    prompts_hash_id_map[prompt_key] = value.hash
                processed_hyperparameters[key] = PromptApi(
                    id=prompts_hash_id_map[prompt_key],
                    type=value.type,
                )
        else:
            processed_hyperparameters[key] = str(value)

    return processed_hyperparameters


def log_hyperparameters(
    func: Optional[Callable] = None,
    *,
    model: Optional[str] = None,
    prompt_template: Optional[str] = None,
):
    def register(f: Callable):
        merged_hyperparameters = f()

        if model is not None or prompt_template is not None:
            if merged_hyperparameters is None:
                merged_hyperparameters = {}

            if isinstance(merged_hyperparameters, dict):
                merged_hyperparameters = dict(merged_hyperparameters)
                if model is not None:
                    merged_hyperparameters["model"] = model
                if prompt_template is not None:
                    merged_hyperparameters["prompt template"] = prompt_template

        test_run = global_test_run_manager.get_test_run()
        test_run.hyperparameters = process_hyperparameters(
            merged_hyperparameters
        )
        global_test_run_manager.save_test_run(TEMP_FILE_PATH)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return register
    return register(func)


def process_prompts(
    hyperparameters: Dict[str, Union[str, int, float, Prompt]],
) -> List[PromptData]:
    prompts = []
    if not hyperparameters:
        return prompts
    seen_prompts = set()
    prompt_objects = [
        value for value in hyperparameters.values() if isinstance(value, Prompt)
    ]
    for prompt in prompt_objects:
        prompt_hash = prompt.hash if is_confident() else None
        prompt_key = f"{prompt.alias}_{prompt_hash}"
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        prompt_data = PromptData(
            alias=prompt.alias,
            hash=prompt_hash,
            version=prompt.version,
            text_template=prompt.text_template,
            messages_template=prompt.messages_template,
            model_settings=prompt.model_settings,
            output_type=prompt.output_type,
            interpolation_type=prompt.interpolation_type,
        )
        prompts.append(prompt_data)
    return prompts
