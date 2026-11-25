import asyncio
import importlib.util
import os
from typing import Optional

from ..schema import CodeFindingsList
from .base import OUTPUT_FORMAT_INSTRUCTION, extract_findings, missing_sdk


class CursorEngine:
    """
    Delegates scanning to the Cursor SDK (uses CURSOR_API_KEY).
    """

    def __init__(self, model: Optional[str] = None):
        if importlib.util.find_spec("cursor_sdk") is None:
            raise missing_sdk("cursor")
        self.model = model

    def generate_findings(self, prompt: str) -> CodeFindingsList:
        from cursor_sdk import Agent

        options = {"local": {"cwd": os.getcwd()}}
        if self.model:
            options["model"] = self.model
        result = Agent.prompt(prompt + OUTPUT_FORMAT_INSTRUCTION, options)
        return extract_findings(result.result)

    async def a_generate_findings(self, prompt: str) -> CodeFindingsList:
        return await asyncio.to_thread(self.generate_findings, prompt)
