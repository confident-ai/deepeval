import logging
from typing import Any


def trace_langchain():
    try:
        from wrapt import wrap_function_wrapper
        from deepeval.integrations.langchain.callback import (
            LangChainCallbackHandler,
        )
        from deepeval.integrations.langchain import _BaseCallbackManagerInit

        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(LangChainCallbackHandler()),
        )
        logging.info("Langchain tracing setup completed.")
    except Exception as e:
        logging.error(f"Error setting up Langchain tracing: {e}")


def trace_llama_index():
    try:
        from deepeval.integrations.llama_index.callback import (
            LlamaIndexCallbackHandler,
        )
        import llama_index.core

        llama_index.core.global_handler = LlamaIndexCallbackHandler()
        logging.info("Llama Index tracing setup completed.")
    except Exception as e:
        logging.error(f"Error setting up Llama Index tracing: {e}")
