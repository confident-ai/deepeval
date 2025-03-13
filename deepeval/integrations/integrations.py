from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from typing import Optional, Sequence, List, Iterable, Any
from itertools import cycle
from enum import Enum
import logging


class Frameworks(Enum):
    LLAMAINDEX = "llama_index"
    LANGCHAIN = "langchain"


captured_data = {}
auto_eval_state = {Frameworks.LANGCHAIN: False, Frameworks.LLAMAINDEX: False}


def trace_langchain(auto_eval: bool = False):

    try:
        from wrapt import wrap_function_wrapper
        from deepeval.integrations.langchain.callback import (
            LangChainCallbackHandler,
        )
        from deepeval.integrations.langchain import _BaseCallbackManagerInit

        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(
                LangChainCallbackHandler(auto_eval=auto_eval)
            ),
        )
        logging.info("Langchain tracing setup completed.")
    except Exception as e:
        logging.error(f"Error setting up Langchain tracing: {e}")

    if auto_eval:

        if any(
            state
            for key, state in auto_eval_state.items()
            if key != Frameworks.LANGCHAIN
        ):
            logging.error(
                "trace_langchain: Another trace function has already been called with auto_eval=True"
            )
            return

        auto_eval_state[Frameworks.LANGCHAIN] = True

        def get_all_subclasses(cls):
            subclasses = cls.__subclasses__()
            for subclass in subclasses:
                subclasses += get_all_subclasses(subclass)
            return subclasses

        # Get all subclasses of VectorStore
        subclasses = get_all_subclasses(VectorStore)

        for subclass in subclasses:
            original_add_documents = getattr(subclass, "add_documents", None)
            original_add_texts = getattr(subclass, "add_texts", None)

            # Monkey patch the add_documents and add_texts method
            if original_add_documents is not None:

                def new_add_documents(
                    self, documents: List[Document], **kwargs
                ) -> List[str]:
                    captured_data[f"{subclass.__name__}_documents"] = documents
                    if captured_data.get("documents") is None:
                        captured_data["documents"] = [documents]
                    else:
                        captured_data["documents"].append(documents)
                    return original_add_documents(self, documents, **kwargs)

                setattr(subclass, "add_documents", new_add_documents)

            if original_add_texts is not None:

                def new_add_texts(
                    self,
                    texts: Iterable[str],
                    metadatas: Optional[List[dict]] = None,
                    **kwargs: Any,
                ) -> List[str]:
                    metadatas_ = iter(metadatas) if metadatas else cycle([{}])
                    documents = [
                        Document(page_content=text, metadata=metadata_)
                        for text, metadata_ in zip(texts, metadatas_)
                    ]
                    captured_data[f"{subclass.__name__}_documents"] = documents
                    if captured_data.get("documents") is None:
                        captured_data["documents"] = [documents]
                    else:
                        captured_data["documents"].append(documents)
                    return original_add_texts(self, texts, metadatas, **kwargs)

                setattr(subclass, "add_texts", new_add_texts)


def trace_llama_index(auto_eval: bool = False):

    try:
        from deepeval.integrations.llama_index.callback import (
            LlamaIndexCallbackHandler,
        )
        import llama_index.core

        llama_index.core.global_handler = LlamaIndexCallbackHandler(
            auto_eval=auto_eval
        )
        logging.info("LlamaIndex `auto_evaluate` setup completed.")
    except Exception as e:
        logging.error(f"Error setting up LlamaIndex `auto_evaluate`: {e}")

    if auto_eval:

        if any(
            state
            for key, state in auto_eval_state.items()
            if key != Frameworks.LLAMAINDEX
        ):
            logging.error(
                "trace_llama_index: Another trace function has already been called with auto_eval=True"
            )
            return

        auto_eval_state[Frameworks.LLAMAINDEX] = True

        # Store original methods
        # Note: capturing base_index and query_engine instead of documents because documents can be added in multiple ways
        original_base_index_init = BaseIndex.__init__
        original_query_engine_init = BaseQueryEngine.__init__

        # Define the patched methods
        def mock_base_index_init(self, *args, **kwargs) -> None:
            captured_data["base_index"] = self
            if "nodes" in kwargs:
                nodes = kwargs.get("nodes")
            elif len(args) > 0 and isinstance(
                args[0], Optional[Sequence[BaseNode]]
            ):
                nodes = args[0]
            captured_data["์nodes"] = nodes
            original_base_index_init(self, *args, **kwargs)

        def mock_query_engine_init(self, *args, **kwargs):
            captured_data["query_engine"] = self
            original_query_engine_init(self, *args, **kwargs)

        # Apply the monkey patch
        BaseIndex.__init__ = mock_base_index_init
        BaseQueryEngine.__init__ = mock_query_engine_init
