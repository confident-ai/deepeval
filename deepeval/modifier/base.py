"""Providing modifier abstractions
"""
from typing import Any, Callable
from abc import abstractmethod


class Modifier:
    """Providing modifiers."""

    @abstractmethod
    def modify(self, input: Any):
        self.input = input


class ClipInterrogatorModifier(Modifier):
    """WIP - Modifies a query using the Clip Interrogator."""

    def __init__(self, completion_function):
        self.completion_function = completion_function

    def clip_interrogate(self, image_path: str, model_name="ViT-L-14/openai"):
        from PIL import Image
        from clip_interrogator import Config, Interrogator

        image = Image.open(image_path).convert("RGB")
        ci = Interrogator(Config(clip_model_name=model_name))
        return ci.interrogate(image)

    def refine_prompt(self, prompt: str, image_path: str = "images/sf_1.jpg"):
        print("refining")
        similar_query = self.clip_interrogate(image_path)
        print("Feeding through to chatgpt")
        new_prompt = self.completion_function(
            f"""Extract the artistic style of the prompt below and modify the original prompt to make it more similar to the prompt below.

    Artistic style to copy: {prompt}
    Original prompt: {similar_query}

    New Prompt:"""
        )
        print("new prompt: ")
        print(new_prompt)
        return new_prompt
