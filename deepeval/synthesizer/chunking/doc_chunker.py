from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters.base import TextSplitter
from typing import Optional, List, Dict
import os

from deepeval.models.base_model import DeepEvalBaseEmbeddingModel


class DocumentChunker:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
    ):
        from chromadb.api.models.Collection import Collection

        self.source_file: Optional[str] = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: Optional[Collection] = None
        self.sections: Optional[List[LCDocument]] = None

        self.text_splitter: TextSplitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embedder: DeepEvalBaseEmbeddingModel = embedder
        self.mean_embedding: Optional[float] = None

        # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, BaseLoader] = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
        }

    async def a_chunk_doc(self) -> List[LCDocument]:
        # Raise error if chunk_doc is called before load_doc
        if self.sections == None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        import chromadb

        # Create ChromaDB client
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        client = chromadb.PersistentClient(path=f".vector_db/{document_name}")

        try:
            collection = client.get_collection(
                name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
            )
            return collection

        except:
            # Convert raw sections to processed chunks
            collection = client.create_collection(
                name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
            )
            langchain_chunks: List[LCDocument] = (
                self.text_splitter.split_documents(self.sections)
            )
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = await self.embedder.a_embed_texts(contents)
            collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=[{"source_file": self.source_file} for i in contents],
                ids=[str(i) for i in range(len(contents))],
            )
            return collection

    def chunk_doc(self):
        # Raise error if chunk_doc is called before load_doc
        if self.sections == None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        import chromadb

        # Create ChromaDB client
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        client = chromadb.PersistentClient(path=f".vector_db/{document_name}")

        try:
            collection = client.get_collection(
                name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
            )
            return collection

        except:
            # Convert raw sections to processed chunks
            collection = client.create_collection(
                name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
            )
            langchain_chunks: List[LCDocument] = (
                self.text_splitter.split_documents(self.sections)
            )
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = self.embedder.embed_texts(contents)
            collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=[{"source_file": self.source_file} for i in contents],
                ids=[str(i) for i in range(len(contents))],
            )
            return collection

    async def a_load_doc(self, path: str) -> List[LCDocument]:
        # Find appropiate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional[BaseLoader] = self.loader_mapping.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load doc into sections and calculate total character count
        loader = loader(path)
        self.sections = await loader.aload()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def load_doc(self, path: str):
        # Find appropiate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional[BaseLoader] = self.loader_mapping.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load doc into sections and calculate total character count
        loader = loader(path)
        self.sections = loader.load()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def count_tokens(self, chunks: LCDocument):
        counter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        return len(counter.split_documents(chunks))
