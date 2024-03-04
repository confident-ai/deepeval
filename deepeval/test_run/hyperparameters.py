from .test_run import test_run_manager


def log_hyperparameters(model: str, prompt_template: str):
    def decorator(func):
        global _model, _user_prompt_template
        _model = model
        _user_prompt_template = prompt_template

        global _hyperparameters
        _hyperparameters = func()

        test_run = test_run_manager.get_test_run()
        test_run.configurations = func()
        test_run.model = model
        test_run.user_prompt_template = prompt_template
        test_run_manager.save_test_run()

        # Define the wrapper function that will be the actual decorator
        def wrapper(*args, **kwargs):
            # Optional: You can decide if you want to do something else here
            # every time the decorated function is called
            return func(*args, **kwargs)

        # Return the wrapper function to be used as the decorator
        return wrapper

    # Return the decorator itself
    return decorator
