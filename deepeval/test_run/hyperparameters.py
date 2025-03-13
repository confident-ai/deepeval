from typing import Union, Dict

from deepeval.test_run import global_test_run_manager
from deepeval.prompt import Prompt, PromptApi


def process_hyperparameters(
    hyperparameters,
) -> Union[Dict[str, Union[str, PromptApi]], None]:
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
            if value._prompt_version_id is not None:
                processed_hyperparameters[key] = PromptApi(
                    promptVersionId=value._prompt_version_id,
                    template=value.template,
                )
            else:
                processed_hyperparameters[key] = str(value.template)
        else:
            processed_hyperparameters[key] = str(value)

    return processed_hyperparameters


def log_hyperparameters(model: str, prompt_template: str):
    def decorator(func):
        test_run = global_test_run_manager.get_test_run()

        def modified_hyperparameters():
            base_hyperparameters = func()
            base_hyperparameters["model"] = model
            base_hyperparameters["prompt template"] = prompt_template
            return base_hyperparameters

        hyperparameters = process_hyperparameters(modified_hyperparameters())
        test_run.hyperparameters = hyperparameters
        global_test_run_manager.save_test_run()

        # Define the wrapper function that will be the actual decorator
        def wrapper(*args, **kwargs):
            # Optional: You can decide if you want to do something else here
            # every time the decorated function is called
            return func(*args, **kwargs)

        # Return the wrapper function to be used as the decorator
        return wrapper

    # Return the decorator itself
    return decorator
