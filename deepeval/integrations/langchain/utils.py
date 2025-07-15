from typing import Any, List, Dict
from langchain_core.outputs import ChatGeneration


def parse_prompts_to_messages(
    prompts: list[str], **kwargs
) -> List[Dict[str, str]]:
    messages = []
    for prompt in prompts:
        for message in prompt.split("\n"):
            if ": " in message:
                role, message = message.split(": ", 1)
                messages.append(
                    {"role": role.strip(), "content": message.strip()}
                )
            else:
                messages.append({"role": "Human", "content": message.strip()})

        tools = kwargs.get("invocation_params", {}).get("tools", None)
        if tools and isinstance(tools, list):
            for tool in tools:
                messages.append({"role": "Tool Input", "content": str(tool)})

    return messages


def convert_chat_generation_to_string(gen: ChatGeneration) -> str:
    return gen.message.pretty_repr()


def prepare_dict(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def extract_token_usage(response_metadata: dict[str, Any]) -> tuple[int, int]:
    if "token_usage" in response_metadata:
        return response_metadata["token_usage"].get(
            "prompt_tokens", 0
        ), response_metadata["token_usage"].get("completion_tokens", 0)
    return 0, 0


def extract_name(serialized: dict[str, Any], **kwargs: Any) -> str:
    if "name" in kwargs and kwargs["name"]:
        return kwargs["name"]

    if "name" in serialized:
        return serialized["name"]

    return "unknown"
