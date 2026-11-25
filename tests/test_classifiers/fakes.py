import json

from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM


class ReplyModel(DeepEvalBaseLLM):
    """A judge that answers every prompt with ``reply`` and records the
    prompts it was sent. Subclasses override ``respond`` to script answers."""

    def __init__(self, reply=None):
        self.reply = (
            {"label": "pricing", "reason": "r"} if reply is None else reply
        )
        self.prompts = []

    def load_model(self):
        return self

    def respond(self, prompt):
        return self.reply

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        response = self.respond(prompt)
        return response if isinstance(response, str) else json.dumps(response)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "reply-model"


class WordEmbedding(DeepEvalBaseEmbeddingModel):
    VOCAB = ["refund", "money", "back", "upgrade", "plan", "slow", "reply"]

    def load_model(self):
        return self

    def embed_text(self, text):
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [1.0 if word in text.lower() else 0.0 for word in self.VOCAB]
            for text in texts
        ]

    async def a_embed_text(self, text):
        return self.embed_text(text)

    async def a_embed_texts(self, texts):
        return self.embed_texts(texts)

    def get_model_name(self):
        return "words"
