"""Module for the query generator
"""
from abc import ABC, abstractmethod
from typing import List


class QueryGenerator(ABC):
    @abstractmethod
    def generate_queries(
        self, texts: List, tags: List = None, num_queries: int = 3
    ):
        pass


class BEIRQueryGenerator:
    """BEIR Query Generator"""

    def __init__(self):
        self._load_model()

    def generate_queries(
        self, texts: List, tags: List = None, num_queries: int = 3
    ):
        """Generates a number of queries"""
        import torch

        queries = []
        for t in texts:
            input_ids = self.tokenizer.encode(t, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=64,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=num_queries,
                )

        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            queries.append(query)
        return queries

    def _load_model(self):
        if hasattr(self, "model"):
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ModuleNotFoundError as e:
            raise Exception("Run `pip install transformers`")

        self.tokenizer = T5Tokenizer.from_pretrained(
            "BeIR/query-gen-msmarco-t5-large-v1"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            "BeIR/query-gen-msmarco-t5-large-v1"
        )
        self.model.eval()
