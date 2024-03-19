from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters.base import TextSplitter

from deepeval.models.openai_embedding_model import OpenAIEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from typing import Optional, List, Tuple, Dict, Type
from pydantic import Field
from enum import Enum
import numpy as np
import uuid
import os


class Chunk:
    def __init__(self):
        self.id = Field(default_factory=lambda: str(uuid.uuid4()))
        self.content = None
        self.embedding = None
        self.source_doc_id = None
        self.similarity_to_mean = None

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
        self.doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: Optional[List[Chunk]] = None

        self.text_splitter: TextSplitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder: DeepEvalBaseEmbeddingModel = embedder
        self.mean_embedding: Optional[float] = None

         # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, Type] = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader
        }

    ############### Load and Chunk ###############
    def load_doc(self, path: str) -> List[LCDocument]:

        # Find appropiate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader_class = self.loader_mapping.get(extension)
        if loader_class is None:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Load and split text in doc
        loader = loader_class(path)
        raw_chunks: List[LCDocument] = loader.load_and_split(self.text_splitter)

        # Load results into Chunk class
        contents = [rc.page_content for rc in raw_chunks] 
        embeddings = self.embedder.embed_documents(contents)
        embeddings_np = np.array(embeddings)
        mean_embedding = np.mean(embeddings_np, axis=0)
        chunks = []
        for i in range(len(raw_chunks)):
            chunk = Chunk()
            chunk.content = contents[i]
            chunk.embedding = embeddings[i]
            chunk.source_doc_id = self.doc_id,
            chunk.similarity_to_mean = get_embedding_similarity((embeddings_np[i]), mean_embedding)
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
       