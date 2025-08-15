import re
from deepeval.tracing import observe, trace_manager


def masking_function(data):
    if isinstance(data, str):
        data = re.sub(r"\b(?:\d{4}[- ]?){3}\d{4}\b", "[REDACTED CARD]", data)
        return data
    return data


trace_manager.configure(mask=masking_function)


@observe()
def llm_app(query: str):
    return "4242-4242-4242-4242"


llm_app("Test Masking")
