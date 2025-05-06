from deepeval.prompt.api import PromptInterpolationType
import re


def interpolate_mustache(text: str, **kwargs) -> str:
    """Interpolate using Mustache format: {{variable}}"""
    formatted_template = re.sub(r"\{\{(\w+)\}\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_mustache_with_space(text: str, **kwargs) -> str:
    """Interpolate using Mustache with space format: {{ variable }}"""
    formatted_template = re.sub(r"\{\{ (\w+) \}\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_fstring(text: str, **kwargs) -> str:
    """Interpolate using F-string format: {variable}"""
    return text.format(**kwargs)


def interpolate_dollar_brackets(text: str, **kwargs) -> str:
    """Interpolate using Dollar Brackets format: ${variable}"""
    formatted_template = re.sub(r"\$\{(\w+)\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_text(
    interpolation_type: PromptInterpolationType, text: str, **kwargs
) -> str:
    """Apply the appropriate interpolation method based on the type"""
    if interpolation_type == PromptInterpolationType.MUSTACHE:
        return interpolate_mustache(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.MUSTACHE_WITH_SPACE:
        return interpolate_mustache_with_space(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.FSTRING:
        return interpolate_fstring(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.DOLLAR_BRACKETS:
        return interpolate_dollar_brackets(text, **kwargs)

    raise ValueError(f"Unsupported interpolation type: {interpolation_type}")
