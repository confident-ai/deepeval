from typing import Union, Dict

from deepeval.test_run import global_test_run_manager
from deepeval.prompt import Prompt, PromptApi
from deepeval.test_run.test_run import TEMP_FILE_PATH


def process_hyperparameters(
    hyperparameters,
) -> Union[Dict[str, Union[str, int, float, PromptApi]], None]:
    if hyperparameters is None:
        return None

    if not isinstance(hyperparameters, dict):
        raise TypeError("Hyperparameters must be a dictionary or None")

    processed_hyperparameters = {}

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
            if value._prompt_version_id is not None and value._type is not None:
                processed_hyperparameters[key] = PromptApi(
                    id=value._prompt_version_id,
                    type=value._type,
                )
            else:
                raise ValueError(
                    f"Cannot log Prompt where template was not pulled from Confident AI. Please import your prompt on Confident AI to continue."
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
