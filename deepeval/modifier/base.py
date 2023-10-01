"""Providing modifier abstractions
"""
from typing import Any
from abc import abstractmethod


class Modifier:
    """Providing modifiers."""

    @abstractmethod
    def modify(self, input: Any):
        self.input = input


class ClipInterrogatorModifier(Modifier):
    def modify(self, input: str):
        from PIL import Image
        from clip_interrogator import Config, Interrogator

        image = Image.open(image_path).convert("RGB")
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        return ci.interrogate(image)
