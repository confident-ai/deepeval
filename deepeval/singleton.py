class Singleton(type):
    """
    Singleton class for having single instance of Metric class.
    This ensures that models aren't loaded twice.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
