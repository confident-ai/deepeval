from abc import abstractmethod
from typing import (
    AsyncIterable,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Protocol,
)

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LCDocument


class FileHandler(Protocol):
    """Interface for custom file type handlers."""

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def load(
        self, path: str, encoding: Optional[str] = None
    ) -> Iterator[LCDocument]:
        """Load a document and return a list of LangChain Document objects."""
        ...

    @abstractmethod
    async def aload(
        self, path: str, encoding: Optional[str] = None
    ) -> AsyncIterator[LCDocument]:
        """Asynchronously load a document and return a list of LangChain Document objects."""
        ...


class FileHandlerLoaderAdapter(BaseLoader):
    """Adapter to make FileHandler compatible with BaseLoader interface."""

    def __init__(
        self, handler: FileHandler, path: str, encoding: Optional[str] = None
    ):
        self.handler = handler
        self.path = path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[LCDocument]:
        return self.handler.load(self.path, self.encoding)
