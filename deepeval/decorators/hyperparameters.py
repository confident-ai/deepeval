_hyperparameters = None


def set_hyperparameters(func):
    global _hyperparameters
    _hyperparameters = func()
    return func


def get_hyperparameters():
    return _hyperparameters
