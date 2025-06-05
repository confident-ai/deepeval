from openai import OpenAI
import functools

from deepeval.tracing.attributes import LlmAttributes
from deepeval.tracing.context import update_current_span
from deepeval.tracing.context import current_span_context
from deepeval.tracing.types import LlmSpan


def patch_openai_client(client: OpenAI):

    original_methods = {}

    # patches these methods
    methods_to_patch = [
        "chat.completions.create",
        "beta.chat.completions.parse",
    ]

    for method_path in methods_to_patch:
        # Split the path into components
        parts = method_path.split(".")
        current_obj = client

        # Navigate to the parent object
        for part in parts[:-1]:
            if not hasattr(current_obj, part):
                print(f"Warning: Cannot find {part} in the path {method_path}")
                continue
            current_obj = getattr(current_obj, part)

        method_name = parts[-1]
        if not hasattr(current_obj, method_name):
            print(
                f"Warning: Cannot find method {method_name} in the path {method_path}"
            )
            continue

        method = getattr(current_obj, method_name)

        if callable(method) and not isinstance(method, type):
            original_methods[method_path] = method

            # Capture the current 'method' using a default argument
            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):
                current_span = current_span_context.get()
                # call the original method using the captured default argument
                response = original_method(*args, **kwargs)
                if isinstance(current_span, LlmSpan):
                    # extract model
                    model = kwargs.get("model", None)
                    if model is None:
                        raise ValueError("model not found in client")

                    # set model
                    current_span.model = model

                    # extract output message
                    output = None
                    try:
                        output = response.choices[0].message.content
                    except Exception as e:
                        pass

                    # extract input output token counts
                    input_token_count = None
                    output_token_count = None
                    try:
                        input_token_count = response.usage.prompt_tokens
                        output_token_count = response.usage.completion_tokens
                    except Exception as e:
                        pass

                    update_current_span(
                        attributes=LlmAttributes(
                            input=kwargs.get(
                                "messages", "INPUT_MESSAGE_NOT_FOUND"
                            ),
                            output=(
                                output if output else "OUTPUT_MESSAGE_NOT_FOUND"
                            ),
                            input_token_count=input_token_count,
                            output_token_count=output_token_count,
                        )
                    )
                return response

            setattr(current_obj, method_name, wrapped_method)
