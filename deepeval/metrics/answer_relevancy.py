from .metric import Metric

class AnswerRelevancy(Metric):
    def __init__(self, success_threshold: bool=0.5):
        self.success_threshold = success_threshold
        from sentence_transformers import CrossEncoder
        self.encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

    def measure(self, query, answer: str) -> float:
        score = self.encoder.predict([query, answer])
        self.success = score > self.success_threshold
        return score

    def is_successful(self) -> bool:
        return self.success
    
    def __name__(self):
        return "Answer Relevancy"
