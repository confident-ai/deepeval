from typing import List, Tuple, Dict, Optional
import random
import asyncio
import math
from tqdm import tqdm as tqdm_bar

from deepeval.synthesizer.chunking.doc_chunker import (
    DocumentChunker,
    Chunk,
    get_embedding_similarity,
)
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

        # TODO: Potential bug, calling generate_goldens_from_docs
        # twice in a notebook enviornment will not refresh source_files_to_chunks_map
        self.doc_to_chunker_map: Optional[Dict[str, List[DocumentChunker]]] = (
            None
        )
        self.source_files_to_chunks_map: Optional[
            Dict[str, List[DocumentChunker]]
        ] = None

    ############### Generate Topics ########################
    def generate_contexts(
        self, num_context_per_document: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        self.check_if_docs_are_loaded()
        contexts: List[List[str]] = []
        source_files: List[str] = []

        # Check if chunk_size is valid for document lengths
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

        # Chunk docs
        self.source_files_to_chunks_map = {}
        for key, chunker in self.doc_to_chunker_map.items():
            self.source_files_to_chunks_map[key] = chunker.chunk_doc()

        # Generate contexts
        for path, document_chunks in tqdm_bar(
            self.source_files_to_chunks_map.items(), desc="Processing files"
        ):
            num_chunks = len(document_chunks)

            min_num_context = min(num_context_per_document, num_chunks)
            clusters = self._get_n_random_clusters(
                path=path, n=min_num_context, cluster_size=max_context_size
            )
            for cluster in clusters:
                context = [chunk.content for chunk in cluster]
                contexts.append(context)
                source_files.append(cluster[0].source_file)

        return contexts, source_files

    ############### Load Docs #############################
    async def _a_load_docs(self) -> Dict[str, List[Chunk]]:
        doc_to_chunker_map = {}

        async def a_process_document(path):
            doc_chunker = DocumentChunker(
                self.embedder, self.chunk_size, self.chunk_overlap
            )
            await doc_chunker.a_load_doc(path)
            if path not in doc_to_chunker_map:
                doc_to_chunker_map[path] = doc_chunker

        tasks = [a_process_document(path) for path in self.document_paths]
        await asyncio.gather(*tasks)
        self.doc_to_chunker_map = doc_to_chunker_map

    def _load_docs(self) -> Dict[str, List[Chunk]]:
        doc_to_chunker_map = {}
        for path in self.document_paths:
            doc_chunker = DocumentChunker(
                self.embedder, self.chunk_size, self.chunk_overlap
            )
            doc_chunker.load_doc(path)
            if path not in doc_to_chunker_map:
                doc_to_chunker_map[path] = doc_chunker
        self.doc_to_chunker_map = doc_to_chunker_map

    ############### Search N Chunks ########################
    def _get_n_random_clusters(
        self, path: str, n: int, cluster_size: int
    ) -> List[List[Chunk]]:
        chunk_cluster = []
        random_chunks = self._get_n_random_chunks(path=path, n=n)
        for chunk in random_chunks:
            cluster = [chunk]
            cluster.extend(
                self._get_n_other_similar_chunks(
                    path=path, query_chunk=chunk, n=cluster_size - 1
                )
            )
            chunk_cluster.append(cluster)
        return chunk_cluster

    def _get_n_random_chunks(self, path: str, n: int) -> List[str]:
        self.check_if_docs_are_loaded()

        document_chunks = self.source_files_to_chunks_map[path]
        n = min(len(document_chunks), n)
        return random.sample(document_chunks, n)

    def _get_n_other_similar_chunks(
        self,
        path: str,
        query_chunk: Chunk,
        n: int,
        threshold: float = 0.7,
    ) -> List[Chunk]:
        self.check_if_docs_are_loaded()

        document_chunks = self.source_files_to_chunks_map[path]

        if not document_chunks:
            raise ValueError("Not enough chunks found for f{path}")

        # Confine n random chunks in case not enough chunks
        n = min(n, len(document_chunks))
        query_embedding = self.embedder.embed_text(query_chunk.content)
        similar_chunks = None

        similarities = []
        filtered_indices = []

        for i, chunk in enumerate(document_chunks):
            sim = get_embedding_similarity(query_embedding, (chunk.embedding))
            similarities.append(sim)

            if sim > threshold and document_chunks[i].id != query_chunk.id:
                filtered_indices.append(i)

        sorted_indices = sorted(
            filtered_indices, key=lambda i: similarities[i], reverse=True
        )
        top_n_indices = sorted_indices[:n]
        similar_chunks = [document_chunks[i] for i in top_n_indices]
        return similar_chunks

    def check_if_docs_are_loaded(self):
        if self.doc_to_chunker_map == None:
            raise ValueError(
                "Context Generator has yet to properly load documents"
            )
