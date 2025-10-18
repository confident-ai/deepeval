import os

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING
from types import SimpleNamespace

from deepeval.models.base_model import DeepEvalBaseEmbeddingModel


if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from langchain_core.documents import Document as LCDocument
    from langchain_text_splitters.base import TextSplitter
    from langchain_community.document_loaders.base import BaseLoader


# Lazy import caches
_langchain_ns = None
_chroma_mod = None
_langchain_import_error = None
_chroma_import_error = None


def _get_langchain():
    """Return a namespace of langchain classes, or raise ImportError with root cause."""
    global _langchain_ns, _langchain_import_error
    if _langchain_ns is not None:
        return _langchain_ns
    try:
        from langchain_core.documents import Document as LCDocument  # type: ignore
        from langchain_text_splitters import TokenTextSplitter  # type: ignore
        from langchain_text_splitters.base import TextSplitter  # type: ignore
        from langchain_community.document_loaders import (  # type: ignore
            PyPDFLoader,
            TextLoader,
            Docx2txtLoader,
        )
        from langchain_community.document_loaders.base import BaseLoader  # type: ignore

        _langchain_ns = SimpleNamespace(
            LCDocument=LCDocument,
            TokenTextSplitter=TokenTextSplitter,
            TextSplitter=TextSplitter,
            PyPDFLoader=PyPDFLoader,
            TextLoader=TextLoader,
            Docx2txtLoader=Docx2txtLoader,
            BaseLoader=BaseLoader,
        )
        return _langchain_ns
    except Exception as e:
        _langchain_import_error = e
        raise ImportError(
            f"langchain, langchain_community, and langchain_text_splitters are required. Root cause: {e}"
        )


def get_chromadb():
    """Return the chromadb module, or raise ImportError with root cause."""
    global _chroma_mod, _chroma_import_error
    if _chroma_mod is not None:
        return _chroma_mod
    try:
        import chromadb

        _chroma_mod = chromadb
        return _chroma_mod
    except Exception as e:
        _chroma_import_error = e
        raise ImportError(
            f"chromadb is required for this functionality. Root cause: {e}"
        )


class DocumentChunker:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
    ):
        self.text_token_count: Optional[int] = None  # set later

        self.source_file: Optional[str] = None
        self.chunks: Optional["Collection"] = None
        self.sections: Optional[List["LCDocument"]] = None
        self.embedder: DeepEvalBaseEmbeddingModel = embedder
        self.mean_embedding: Optional[float] = None

        # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, "Type[BaseLoader]"] = {}

    #########################################################
    ### Chunking Docs #######################################
    #########################################################

    async def a_chunk_doc(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> "Collection":
        lc = _get_langchain()
        chroma = get_chromadb()

        from chromadb.config import Settings as ChromaSettings

        # Raise error if chunk_doc is called before load_doc
        if self.sections is None or self.source_file is None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        # Determine client and collection_name
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        if client is None:
            client = chroma.PersistentClient(
                path=f".vector_db/{document_name}",
                settings=ChromaSettings(anonymized_telemetry=True),
            )
            default_coll = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        else:
            # namespace by doc to support sharing a single client across many docs
            default_coll = (
                f"{document_name}_processed_chunks_{chunk_size}_{chunk_overlap}"
            )
        collection_name = collection_name or default_coll

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            text_splitter: "TextSplitter" = lc.TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)

            langchain_chunks = text_splitter.split_documents(self.sections)
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = await self.embedder.a_embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas: List[dict] = [
                    {"source_file": self.source_file} for _ in batch_contents
                ]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
        return collection

    def chunk_doc(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ):
        lc = _get_langchain()
        chroma = get_chromadb()

        from chromadb.config import Settings as ChromaSettings

        # Raise error if chunk_doc is called before load_doc
        if self.sections is None or self.source_file is None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        # Determine client and collection_name
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        if client is None:
            client = chroma.PersistentClient(
                path=f".vector_db/{document_name}",
                settings=ChromaSettings(anonymized_telemetry=True),
            )
            default_coll = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        else:
            # namespace by doc to support sharing a single client across many docs
            default_coll = (
                f"{document_name}_processed_chunks_{chunk_size}_{chunk_overlap}"
            )
        collection_name = collection_name or default_coll

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            text_splitter: "TextSplitter" = lc.TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)

            langchain_chunks = text_splitter.split_documents(self.sections)
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = self.embedder.embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas: List[dict] = [
                    {"source_file": self.source_file} for _ in batch_contents
                ]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
        return collection

    #########################################################
    ### Loading Docs ########################################
    #########################################################

    def get_loader(self, path: str, encoding: Optional[str]) -> "BaseLoader":
        lc = _get_langchain()
        # set mapping lazily now that langchain classes exist
        if not self.loader_mapping:
            self.loader_mapping = {
                ".pdf": lc.PyPDFLoader,
                ".txt": lc.TextLoader,
                ".docx": lc.Docx2txtLoader,
                ".md": lc.TextLoader,
                ".markdown": lc.TextLoader,
                ".mdx": lc.TextLoader,
            }

        # Find appropriate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional["Type[BaseLoader]"] = self.loader_mapping.get(
            extension
        )
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load doc into sections and calculate total token count
        if loader is lc.TextLoader:
            return loader(path, encoding=encoding, autodetect_encoding=True)
        elif loader in (lc.PyPDFLoader, lc.Docx2txtLoader):
            return loader(path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    async def a_load_doc(self, path: str, encoding: Optional[str]):
        loader = self.get_loader(path, encoding)
        self.sections = await loader.aload()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def load_doc(self, path: str, encoding: Optional[str]):
        loader = self.get_loader(path, encoding)
        self.sections = loader.load()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def count_tokens(self, chunks: List["LCDocument"]):
        lc = _get_langchain()
        counter = lc.TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        return len(counter.split_documents(chunks))
