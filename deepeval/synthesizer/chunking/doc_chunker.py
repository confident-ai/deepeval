from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters.base import TextSplitter
from typing import Optional, List, Dict, Union

from llama_index.core.schema import TextNode
import os

from deepeval.models.base_model import DeepEvalBaseEmbeddingModel


class DocumentChunker:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
    ):
        from chromadb.api.models.Collection import Collection

        self.source_file: Optional[str] = None
        self.chunks: Optional[Collection] = None
        self.sections: Optional[List[LCDocument]] = None
        self.embedder: DeepEvalBaseEmbeddingModel = embedder
        self.mean_embedding: Optional[float] = None

        # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, BaseLoader] = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
        }

    #########################################################
    ### Chunking Docs #######################################
    #########################################################

    async def a_chunk_doc(
        self, chunk_size: int = 1024, chunk_overlap: int = 0
    ) -> List[LCDocument]:
        text_splitter: TextSplitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
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

        collection_name = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
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
                batch_metadatas = [
                    {"source_file": self.source_file} for _ in batch_contents
                ]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
        return collection

    def chunk_doc(self, chunk_size: int = 1024, chunk_overlap: int = 0):
        text_splitter: TextSplitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
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

        collection_name = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
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
                batch_metadatas = [
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
    ### Create collection from node #########################
    #########################################################

    async def a_from_nodes(self, nodes: List[Union[TextNode, LCDocument]]):
        # Create ChromaDB client
        import chromadb

        client = chromadb.PersistentClient(path=f".vector_db/{nodes[0].id_}")
        collection_name = "processed_chunks"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)

            contents = []
            source_files = []
            for node in nodes:
                if isinstance(node, TextNode):
                    contents.append(node.text)
                    source_files.append(
                        {"source_file": node.metadata.get("", "None")}
                    )
                elif isinstance(node, LCDocument):
                    contents.append(node.page_content)
                    source_files.append({"source_file": "None"})

            embeddings = await self.embedder.a_embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_medata = source_files[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_medata,
                    ids=batch_ids,
                )
        return collection

    def from_nodes(self, nodes: List[Union[TextNode, LCDocument]]):
        # Create ChromaDB client
        import chromadb

        client = chromadb.PersistentClient(path=f".vector_db/{nodes[0].id_}")
        collection_name = "processed_chunks"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)
            contents = [node.text for node in nodes]
            source_files = [
                {"source_file": node.metadata.get("", "None")} for node in nodes
            ]
            embeddings = self.embedder.embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_medata = source_files[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_medata,
                    ids=batch_ids,
                )
        return collection

    #########################################################
    ### Loading Docs ########################################
    #########################################################

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
