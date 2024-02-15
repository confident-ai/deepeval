_hyperparameters = None
_model = None


def set_hyperparameters(model: str):
    def decorator(func):
        global _model
        _model = model

        global _hyperparameters
        _hyperparameters = func()

        # Define the wrapper function that will be the actual decorator
        def wrapper(*args, **kwargs):
            # Optional: You can decide if you want to do something else here
            # every time the decorated function is called
            return func(*args, **kwargs)

        # Return the wrapper function to be used as the decorator
        return wrapper

    # Return the decorator itself
    return decorator


def get_hyperparameters():
    return _hyperparameters


def get_model():
    return _model
