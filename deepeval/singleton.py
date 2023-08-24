class Singleton(type):
    """
    Singleton class for having a single instance of a class.
    This ensures that instances aren't created more than once.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)
