from typing import Union, Dict, Optional, List
from deepeval.test_run import global_test_run_manager
from deepeval.prompt import Prompt
from deepeval.prompt.api import PromptApi
from deepeval.test_run.test_run import TEMP_FILE_PATH
from deepeval.confident.api import is_confident
from deepeval.test_run.test_run import PromptData


def process_hyperparameters(
    hyperparameters: Optional[Dict] = None,
    verbose: bool = True,
) -> Union[Dict[str, Union[str, int, float, PromptApi]], None]:
    if hyperparameters is None:
        return None

    if not isinstance(hyperparameters, dict):
        raise TypeError("Hyperparameters must be a dictionary or None")

    processed_hyperparameters = {}
    prompts_version_id_map = {}

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
                prompt_key = f"{value.alias}_{value.version}"
            except AttributeError:
                prompt_key = f"{value.alias}_00.00.01"

            if value._prompt_version_id is not None and value.type is not None:
                processed_hyperparameters[key] = PromptApi(
                    id=value._prompt_version_id,
                    type=value.type,
                )
            elif is_confident():
                if prompt_key not in prompts_version_id_map:
                    value.push(_verbose=verbose)
                    prompts_version_id_map[prompt_key] = (
                        value._prompt_version_id
                    )
                processed_hyperparameters[key] = PromptApi(
                    id=prompts_version_id_map[prompt_key],
                    type=value.type,
                )
        else:
            processed_hyperparameters[key] = str(value)

    return processed_hyperparameters


def log_hyperparameters(func):
    test_run = global_test_run_manager.get_test_run()

    def modified_hyperparameters():
        base_hyperparameters = func()
        return base_hyperparameters

    hyperparameters = process_hyperparameters(modified_hyperparameters())
    test_run.hyperparameters = hyperparameters
    global_test_run_manager.save_test_run(TEMP_FILE_PATH)

    # Define the wrapper function that will be the actual decorator
    def wrapper(*args, **kwargs):
        # Optional: You can decide if you want to do something else here
        # every time the decorated function is called
        return func(*args, **kwargs)

    # Return the wrapper function to be used as the decorator
    return wrapper


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
        prompt_version = prompt.version if is_confident() else None
        prompt_key = f"{prompt.alias}_{prompt_version}"
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        prompt_data = PromptData(
            alias=prompt.alias,
            version=prompt_version,
            text_template=prompt.text_template,
            messages_template=prompt.messages_template,
            model_settings=prompt.model_settings,
            output_type=prompt.output_type,
            interpolation_type=prompt.interpolation_type,
        )
        prompts.append(prompt_data)
    return prompts
