import importlib.util
from typing import Optional

from ..schema import CodeFindingsList
from .base import OUTPUT_FORMAT_INSTRUCTION, extract_findings, missing_sdk


class CodexEngine:
    """
    Delegates scanning to the Codex SDK (uses OPENAI_API_KEY).
    """

    def __init__(self, model: Optional[str] = None):
        if importlib.util.find_spec("openai_codex") is None:
            raise missing_sdk("codex")
        self.model = model

    def generate_findings(self, prompt: str) -> CodeFindingsList:
        from openai_codex import Codex, Sandbox

        kwargs = {"sandbox": Sandbox.read_only}
        if self.model:
            kwargs["model"] = self.model
        with Codex() as codex:
            thread = codex.thread_start(**kwargs)
            result = thread.run(prompt + OUTPUT_FORMAT_INSTRUCTION)
        return extract_findings(result.final_response)

    async def a_generate_findings(self, prompt: str) -> CodeFindingsList:
        from openai_codex import AsyncCodex, Sandbox

        kwargs = {"sandbox": Sandbox.read_only}
        if self.model:
            kwargs["model"] = self.model
        async with AsyncCodex() as codex:
            thread = await codex.thread_start(**kwargs)
            result = await thread.run(prompt + OUTPUT_FORMAT_INSTRUCTION)
        return extract_findings(result.final_response)
