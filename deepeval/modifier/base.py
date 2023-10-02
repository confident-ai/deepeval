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
    """Modifies a query"""

    def modify(self, image_path: str, model_name="ViT-L-14/openai"):
        from PIL import Image
        from clip_interrogator import Config, Interrogator

        image = Image.open(image_path).convert("RGB")
        ci = Interrogator(Config(clip_model_name=model_name))
        return ci.interrogate(image)
