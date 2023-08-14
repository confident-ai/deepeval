from .metric import Metric


class BertScore(Metric):
    """basic implementation of BertScore"""

    def measure(self, a: str, b: str):
        raise NotImplementedError
