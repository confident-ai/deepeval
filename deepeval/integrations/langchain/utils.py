from typing import Any, List, Dict
from langchain_core.outputs import ChatGeneration


def parse_prompts_to_messages(
    prompts: list[str], **kwargs
) -> List[Dict[str, str]]:
    VALID_ROLES = [
        "system",
        "assistant",
        "ai",
        "user",
        "human",
        "tool",
        "function",
    ]

    messages: List[Dict[str, str]] = []
    current_role = None
    current_content: List[str] = []

    for prompt in prompts:
        for line in prompt.splitlines():
            line = line.strip()
            if not line:
                continue

            first_word, sep, rest = line.partition(":")
            role = (
                first_word.lower()
                if sep and first_word.lower() in VALID_ROLES
                else None
            )

            if role:
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_role = role
                current_content = [rest.strip()]
            else:
                if not current_role:
                    current_role = "Human"
                current_content.append(line)

        if current_role and current_content:
            messages.append(
                {
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                }
            )
            current_role, current_content = None, []

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
