from pydantic import BaseModel, Field


class TranslationRewriteSchema(BaseModel):
    """Structured response for metric template translation (matches metric LLM patterns)."""

    rewritten_template: str = Field(
        ...,
        description="Full translated template; placeholders like {{name}} unchanged.",
    )
