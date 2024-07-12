import logging

class Integrations:

    @staticmethod
    def trace_langchain():
        try:
            from wrapt import wrap_function_wrapper
            from deepeval.integrations.langchain.callback import LangChainCallbackHandler
            from deepeval.integrations.langchain import _BaseCallbackManagerInit

            wrap_function_wrapper(
                module="langchain_core.callbacks",
                name="BaseCallbackManager.__init__",
                wrapper=_BaseCallbackManagerInit(LangChainCallbackHandler()),
            )
            logging.info("Langchain tracing setup completed.")
        except Exception as e:
            logging.error(f"Error setting up Langchain tracing: {e}")

    @staticmethod
    def trace_llama_index():
        try:
            from llama_index.core import set_global_handler
            set_global_handler("deepeval")
            logging.info("Llama Index tracing setup completed.")
        except Exception as e:
            logging.error(f"Error setting up Llama Index tracing: {e}")