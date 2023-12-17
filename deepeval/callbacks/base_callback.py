from abc import ABC, abstractmethod

class BaseCallback(ABC):
    """
    Base class for training callbacks in deepeval.

    Attributes:
        - metrics (list): List of metrics to be evaluated.
        - evaluate_every (int): Frequency of metric evaluation during training.
    """

    def __init__(self, metrics=None, evaluate_every=1):
        """
        Initialize the BaseCallback.

        Args:
            metrics (list, optional): List of metrics to be evaluated.
            evaluate_every (int, optional): Frequency of metric evaluation during training.
        """
        self.metrics = metrics or []
        self.evaluate_every = evaluate_every

    @abstractmethod
    def on_epoch_begin(self, trainer, epoch, logs=None):
        """
        Called at the beginning of each epoch.

        Args:
            trainer: The training framework's trainer object.
            epoch (int): Current epoch.
            logs (dict, optional): Dictionary to store additional information.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch, logs=None):
        """
        Called at the end of each epoch.

        Args:
            trainer: The training framework's trainer object.
            epoch (int): Current epoch.
            logs (dict, optional): Dictionary to store additional information.
        """
        pass

    @abstractmethod
    def on_batch_begin(self, trainer, batch, logs=None):
        """
        Called at the beginning of each batch.

        Args:
            trainer: The training framework's trainer object.
            batch: Current batch.
            logs (dict, optional): Dictionary to store additional information.
        """
        pass

    @abstractmethod
    def on_batch_end(self, trainer, batch, logs=None):
        """
        Called at the end of each batch.

        Args:
            trainer: The training framework's trainer object.
            batch: Current batch.
            logs (dict, optional): Dictionary to store additional information.
        """
        pass

    def evaluate_metrics(self, trainer):
        """
        Evaluate metrics based on the specified frequency.

        Args:
            trainer: The training framework's trainer object.

        Returns:
            dict: Dictionary containing metric results.
        """
        pass

    @abstractmethod
    def compute_metric(self, trainer, metric):
        """
        Compute the value of a specific metric.

        Args:
            trainer: The training framework's trainer object.
            metric (str): The metric to be computed.

        Returns:
            float: Computed metric value.
        """
        pass

    def log_metrics(self, epoch, metrics_results):
        """
        Log the evaluated metrics.

        Args:
            epoch (int): Current epoch.
            metrics_results (dict): Dictionary containing metric results.
        """
        pass
