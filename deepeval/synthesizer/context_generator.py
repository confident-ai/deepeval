from typing import List, Tuple, Dict, Union, Optional
import random
import asyncio

from deepeval.synthesizer.doc_chunker import (
    DocumentChunker,
    Chunk,
    get_embedding_similarity,
)
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.utils import get_or_create_event_loop


class ContextGenerator:
    def __init__(
        self,
        document_paths: List[str],
        embedder: DeepEvalBaseEmbeddingModel,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        async_mode: bool = False,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.async_mode = async_mode
        self.document_paths: List[str] = document_paths

        # TODO: Potential bug, calling generate_goldens_from_docs
        # twice in a notebook enviornment will not refresh source_files_to_chunks_map
        self.source_files_to_chunks_map: Dict[str, List[Chunk]] = (
            self._load_docs()
        )

    ############### Generate Topics ########################
    def generate_contexts(
        self, num_context: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        contexts: List[List[str]] = []
        source_files: List[str] = []
        for path, document_chunks in self.source_files_to_chunks_map.items():
            num_chunks = len(document_chunks)
            num_context = min(num_context, num_chunks)
            clusters = self._get_n_random_clusters(
                path=path, n=num_context, cluster_size=max_context_size
            )
            for cluster in clusters:
                context = [chunk.content for chunk in cluster]
                contexts.append(context)
                source_files.append(cluster[0].source_file)

        return contexts, source_files

    ############### Load Docs #############################
    async def _a_load_docs(self) -> Dict[str, List[Chunk]]:
        async def a_process_document(path):
            doc_chunker = DocumentChunker(
                self.embedder, self.chunk_size, self.chunk_overlap
            )
            chunks = await doc_chunker.a_load_doc(path)
            return path, chunks

        source_files_to_chunks_map: Dict[str, List[Chunk]] = {}
        tasks = [a_process_document(path) for path in self.document_paths]
        path_to_chunks = await asyncio.gather(*tasks)
        for path, chunks in path_to_chunks:
            if path not in source_files_to_chunks_map:
                source_files_to_chunks_map[path] = []
            source_files_to_chunks_map[path].extend(chunks)
        return source_files_to_chunks_map

    def _load_docs(self) -> Dict[str, List[Chunk]]:
        if not self.async_mode:
            source_files_to_chunks_map: Dict[str, List[Chunk]] = {}
            for path in self.document_paths:
                doc_chunker = DocumentChunker(
                    self.embedder, self.chunk_size, self.chunk_overlap
                )
                chunks = doc_chunker.load_doc(path)
                if path not in source_files_to_chunks_map:
                    source_files_to_chunks_map[path] = []
                source_files_to_chunks_map[path].extend(chunks)
            return source_files_to_chunks_map
        else:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(self._a_load_docs())

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
