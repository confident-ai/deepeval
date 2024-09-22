from chromadb.api.models.Collection import Collection
from typing import List, Tuple, Dict, Optional
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm as tqdm_bar
import chromadb
import asyncio
import random
import math
import os

from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel


class ContextGenerator:
    def __init__(
        self,
        document_paths: List[str],
        embedder: DeepEvalBaseEmbeddingModel,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_paths: List[str] = document_paths
        self.total_chunks = 0

        # TODO: Potential bug, calling generate_goldens_from_docs
        # twice in a notebook enviornment will not refresh source_files_to_chunks_map
        self.doc_to_chunker_map: Optional[Dict[str, DocumentChunker]] = None
        self.source_files_to_collections_map: Optional[
            Dict[str, Collection]
        ] = None

    async def a_generate_contexts(
        self, num_context_per_document: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        self.check_if_docs_are_loaded()
        contexts: List[List[str]] = []
        source_files: List[str] = []

        # Check if chunk_size is valid for document lengths
        if self.doc_to_chunker_map is not None:
            min_doc_token_count = min(
                chunker.text_token_count
                for chunker in self.doc_to_chunker_map.values()
            )
            num_contexts_limit = 1 + math.floor(
                (min_doc_token_count - self.chunk_size)
                / (self.chunk_size - self.chunk_overlap)
            )
            if num_contexts_limit < num_context_per_document:
                suggested_chunk_size = (
                    min_doc_token_count
                    + (self.chunk_overlap * (num_context_per_document - 1))
                ) // num_context_per_document
                raise ValueError(
                    f"Your smallest document is only sized {min_doc_token_count}."
                    f"Please adjust the chunk_size to no more than {suggested_chunk_size}."
                )

        # Chunk docs if not already cached via ChromaDB
        async def a_chunk_and_store(key, chunker: DocumentChunker):
            self.source_files_to_collections_map[key] = (
                await chunker.a_chunk_doc()
            )

        if self.source_files_to_collections_map == None:
            self.source_files_to_collections_map = {}
        if self.doc_to_chunker_map != None:
            tasks = [
                a_chunk_and_store(key, chunker)
                for key, chunker in self.doc_to_chunker_map.items()
            ]
            await tqdm_asyncio.gather(
                *tasks, desc="✨ 📚 ✨ Chunking Documents"
            )

        # Generate contexts
        for path, collection in tqdm_bar(
            self.source_files_to_collections_map.items(),
            desc="✨ 🧩 ✨ Generating Contexts",
        ):
            num_chunks = collection.count()
            min_num_context = min(num_context_per_document, num_chunks)
            contexts.extend(
                self._get_n_random_contexts_per_doc(
                    path=path,
                    n_contexts_per_doc=min_num_context,
                    context_size=max_context_size,
                    similarity_threshold=0.7,
                )
            )
            for _ in contexts:
                source_files.append(path)
            self.total_chunks += num_chunks
        return contexts, source_files

    def generate_contexts(
        self, num_context_per_document: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        self.check_if_docs_are_loaded()
        contexts: List[List[str]] = []
        source_files: List[str] = []

        # Check if chunk_size is valid for document lengths
        if self.doc_to_chunker_map is not None:
            min_doc_token_count = min(
                chunker.text_token_count
                for chunker in self.doc_to_chunker_map.values()
            )
            max_num_contexts_possible = 1 + math.floor(
                (min_doc_token_count - self.chunk_size)
                / (self.chunk_size - self.chunk_overlap)
            )
            if max_num_contexts_possible < num_context_per_document:
                suggested_chunk_size = (
                    min_doc_token_count
                    + (self.chunk_overlap * (num_context_per_document - 1))
                ) // num_context_per_document
                raise ValueError(
                    f"Your smallest document is only sized {min_doc_token_count}. "
                    f"Please adjust the chunk_size to no more than {suggested_chunk_size}."
                )

        # Chunk docs if not already cached via ChromaDB
        if self.source_files_to_collections_map == None:
            self.source_files_to_collections_map = {}
        if self.doc_to_chunker_map != None:
            for key, chunker in tqdm_bar(
                self.doc_to_chunker_map.items(), "✨ 📚 ✨ Chunking Documents"
            ):
                self.source_files_to_collections_map[key] = chunker.chunk_doc()

        # Generate contexts
        for path, collection in tqdm_bar(
            self.source_files_to_collections_map.items(),
            desc="✨ 🧩 ✨ Generating Contexts",
        ):
            num_chunks = collection.count()
            min_num_context = min(num_context_per_document, num_chunks)
            contexts.extend(
                self._get_n_random_contexts_per_doc(
                    path=path,
                    n_contexts_per_doc=min_num_context,
                    context_size=max_context_size,
                    similarity_threshold=0.5,
                )
            )
            for _ in contexts:
                source_files.append(path)
            self.total_chunks += num_chunks
        return contexts, source_files

    async def _a_load_docs(self):
        async def a_process_document(path):
            try:
                # Create ChromaDB client
                full_document_path, _ = os.path.splitext(path)
                document_name = os.path.basename(full_document_path)
                client = chromadb.PersistentClient(
                    path=f".vector_db/{document_name}"
                )
                collection = client.get_collection(
                    name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
                )

                # Needs to strictly be after getting collection so map is assigned to None if exception is raised
                if self.source_files_to_collections_map == None:
                    self.source_files_to_collections_map = {}
                self.source_files_to_collections_map[path] = collection

            except Exception as e:
                if self.doc_to_chunker_map is None:
                    self.doc_to_chunker_map = {}
                doc_chunker = DocumentChunker(
                    self.embedder, self.chunk_size, self.chunk_overlap
                )
                await doc_chunker.a_load_doc(path)
                if path not in self.doc_to_chunker_map:
                    self.doc_to_chunker_map[path] = doc_chunker

        # Process all documents asynchronously with a progress bar
        tasks = [a_process_document(path) for path in self.document_paths]
        await tqdm_asyncio.gather(*tasks, desc="✨ 🚀 ✨ Loading Documents")

    def _load_docs(self):
        for path in tqdm_bar(self.document_paths, "✨ 🚀 ✨ Loading Documents"):
            try:
                # Create ChromaDB client
                full_document_path, _ = os.path.splitext(path)
                document_name = os.path.basename(full_document_path)
                client = chromadb.PersistentClient(
                    path=f".vector_db/{document_name}"
                )
                collection = client.get_collection(
                    name=f"processed_chunks_{self.chunk_size}_{self.chunk_overlap}"
                )

                # Needs to strictly be after getting collection so map is assigned to None if exception is raised
                if self.source_files_to_collections_map == None:
                    self.source_files_to_collections_map = {}
                self.source_files_to_collections_map[path] = collection

            except:
                if self.doc_to_chunker_map == None:
                    self.doc_to_chunker_map = {}
                doc_chunker = DocumentChunker(
                    self.embedder, self.chunk_size, self.chunk_overlap
                )
                doc_chunker.load_doc(path)
                if path not in self.doc_to_chunker_map:
                    self.doc_to_chunker_map[path] = doc_chunker

    def _get_n_random_contexts_per_doc(
        self,
        path: str,
        n_contexts_per_doc: int,
        context_size: int,
        similarity_threshold: int,
    ):
        assert (
            n_contexts_per_doc > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."

        contexts = []
        num_query_docs = 0

        # get [n=n_contexts_per_doc] random chunks per doc
        random_chunks = self._get_n_random_chunks_per_doc(
            path=path, n_chunks=n_contexts_per_doc
        )
        collection = self.source_files_to_collections_map[path]

        # for each random chunk find [n=context_size] similar chunks to form a context
        for i in range(len(random_chunks)):
            random_chunk = random_chunks[i]
            if not random_chunk.strip():
                continue

            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk),
                n_results=min(context_size, collection.count()),
            )
            context = [random_chunk]

            # disregard repeated chunks and chunks that don't pass the similarity threshold
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):
                similar_chunk_similarity = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in similar_chunk_texts
                    and similar_chunk_similarity > similarity_threshold
                ):
                    context.append(similar_chunk_text)
            contexts.append(context)
        return contexts

    def _get_n_random_chunks_per_doc(
        self, path: str, n_chunks: int
    ) -> Tuple[List[str], List[str]]:
        collection = self.source_files_to_collections_map[path]
        total_chunks = collection.count()
        assert (
            n_chunks <= total_chunks
        ), f"Requested {n_chunks} chunks, but the collection only contains {total_chunks} chunks."

        # randomly sample n chunks from the collection
        n_random_ids = [
            str(i) for i in random.sample(range(total_chunks), n_chunks)
        ]
        chunks = collection.get(ids=n_random_ids)
        return chunks["documents"]

    def check_if_docs_are_loaded(self):
        if (
            self.doc_to_chunker_map == None
            and self.source_files_to_collections_map == None
        ):
            raise ValueError(
                "Context Generator has yet to properly load documents"
            )
