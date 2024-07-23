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
from pydantic import BaseModel
import numpy as np
import uuid
import os

from deepeval.models.base_model import DeepEvalBaseEmbeddingModel


class Chunk(BaseModel):
    id: str
    content: str
    embedding: List[float]
    source_file: str
    similarity_to_mean: float


####################################################
############### Main Doc Chunker ###################
####################################################


class DocumentChunker:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
    ):
        self.source_file: Optional[str] = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: Optional[List[Chunk]] = None

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

    ############### Load and Chunk ###############
    async def a_load_doc(self, path: str) -> List[LCDocument]:
        self.source_file = path

        # Find appropiate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional[BaseLoader] = self.loader_mapping.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load and split text in doc
        loader = loader(path)
        docs = await loader.aload()
        raw_chunks: List[LCDocument] = self.text_splitter.split_documents(docs)

        # Load results into Chunk class
        contents = [rc.page_content for rc in raw_chunks]
        embeddings = await self.embedder.a_embed_texts(contents)
        embeddings_np = np.array(embeddings)
        mean_embedding = np.mean(embeddings_np, axis=0)
        chunks = []
        for i in range(len(raw_chunks)):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=contents[i],
                embedding=embeddings[i],
                source_file=path,
                similarity_to_mean=get_embedding_similarity(
                    (embeddings_np[i]), mean_embedding
                ),
            )
            chunks.append(chunk)
        self.chunks = chunks

        return chunks

    ############### Load and Chunk ###############
    def load_doc(self, path: str) -> List[LCDocument]:
        self.source_file = path

        # Find appropiate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional[BaseLoader] = self.loader_mapping.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load and split text in doc
        loader = loader(path)
        raw_chunks: List[LCDocument] = loader.load_and_split(self.text_splitter)

        # Load results into Chunk class
        contents = [rc.page_content for rc in raw_chunks]
        embeddings = self.embedder.embed_texts(contents)
        embeddings_np = np.array(embeddings)
        mean_embedding = np.mean(embeddings_np, axis=0)
        chunks = []
        for i in range(len(raw_chunks)):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=contents[i],
                embedding=embeddings[i],
                source_file=path,
                similarity_to_mean=get_embedding_similarity(
                    (embeddings_np[i]), mean_embedding
                ),
            )
            chunks.append(chunk)
        self.chunks = chunks

        return chunks


####################################################
############### Similarity Funcs ###################
####################################################


def get_embedding_similarity(
    embedding1: List,
    embedding2: List,
) -> float:
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm
