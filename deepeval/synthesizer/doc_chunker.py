from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters.base import TextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import Optional, List, Tuple
from pydantic import Field
from enum import Enum
import numpy as np
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
        chunk_size: int = 1024,
        chunk_overlap: int = 0
    ):
        # Unique ID for each document
        self.doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Choose TokenTextSplitter as only text_splitter for now
        self.text_splitter: TextSplitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Chunks contain LCDocument, while nodes contain strings (LCDocument.page_content)
        self.chunks: Optional[List[Chunk]] = None
        
        # Embedder is the embedding model and embeddings are the embedded nodes
        self.embedder: OpenAIEmbeddings = OpenAIEmbeddings()
        self.mean_embedding: Optional[float] = None

         # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, Type] = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader
        }

    ############### Load and Chunk ###############
    def load_doc(self, path: str) -> List[LCDocument]:

        _, extension = os.path.splitext(path)
        extension = extension.lower()
        # Select the appropriate loader based on the file extension
        loader_class = self.loader_mapping.get(extension)
        if loader_class is None:
            raise ValueError(f"Unsupported file format: {extension}")
        
        loader = loader_class(path)
        raw_chunks = loader.load_and_split(self.text_splitter)
        contents = [rc.page_content for rc in raw_chunks] 

        # Utilize lang_chain for run async
        embeddings = self.embedder.embed_documents(contents)
        embeddings_np = np.array(embeddings)
        mean_embedding = np.mean(embeddings_np, axis=0)

        chunks = []
        for i in range(len(raw_chunks)):
            chunk = Chunk()
            chunk.content = contents[i]
            chunk.embedding = embeddings[i]
            chunk.source_doc_id = self.doc_id,
            chunk.similarity_to_mean = Similarity.get_embedding_similarity((embeddings_np[i]), mean_embedding)
            chunks.append(chunk)
        self.chunks = chunks

        return chunks

####################################################
############### Similarity Funcs ###################
####################################################
    
class SIMILARITY_MEASURE(str, Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"

class Similarity:
    
    @staticmethod
    def get_embedding_similarity(
        embedding1: List,
        embedding2: List,
        type: Enum = SIMILARITY_MEASURE.COSINE,
        ) -> float:
         if type == SIMILARITY_MEASURE.COSINE:
             product = np.dot(embedding1, embedding2)
             norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
             return product / norm
         elif type == SIMILARITY_MEASURE.EUCLIDEAN:
            return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
         elif type == SIMILARITY_MEASURE.DOT_PRODUCT:
                return np.dot(embedding1, embedding2)    