from typing import List, Tuple, Dict, Optional, Union
from langchain_core.documents import Document
from llama_index.core.schema import TextNode
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm as tqdm_bar
from pydantic import BaseModel
import asyncio
import random
import math
import os

from deepeval.models.base_model import (
    DeepEvalBaseEmbeddingModel,
    DeepEvalBaseLLM,
)
from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.synthesizer.templates.template import FilterTemplate


class ContextScore(BaseModel):
    clarity: float
    depth: float
    structure: float
    relevance: float


class ContextGenerator:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
        document_paths: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        max_retries: int = 3,
        filter_threshold: float = 0.5,
        similarity_threshold: float = 0.5,
        _nodes: Optional[List[Union[TextNode, Document]]] = None,
    ):
        from chromadb.api.models.Collection import Collection

        # Ensure either document_paths or _nodes is provided
        if not document_paths and not _nodes:
            raise ValueError("`document_path` is empty or missing.")

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.total_chunks = 0
        self.document_paths: List[str] = document_paths
        self._nodes = _nodes

        # Model parameters
        self.model, self.using_native_model = initialize_model(model)
        self.embedder = embedder

        # Quality parameters
        self.max_retries = max_retries
        self.filter_threshold = filter_threshold
        self.similarity_threshold = similarity_threshold

        # TODO: Potential bug, calling generate_goldens_from_docs
        # twice in a notebook enviornment will not refresh source_files_to_chunks_map
        self.doc_to_chunker_map: Optional[Dict[str, DocumentChunker]] = None
        self.source_files_to_collections_map: Optional[
            Dict[str, Collection]
        ] = None

        # cost tracking
        self.total_cost = 0.0

    #########################################################
    ### Generate Contexts ###################################
    #########################################################

    def generate_contexts(
        self, num_context_per_document: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str], List[float]]:
        self.check_if_docs_are_loaded()
        scores = []
        contexts = []
        source_files = []

        # Check if chunk_size is valid for document lengths
        if self.doc_to_chunker_map is not None:
            smallest_document_token_count = min(
                chunker.text_token_count
                for chunker in self.doc_to_chunker_map.values()
            )
            smallest_document_num_chunks = 1 + math.floor(
                (smallest_document_token_count - self.chunk_size)
                / (self.chunk_size - self.chunk_overlap)
            )
            if smallest_document_num_chunks < num_context_per_document:
                suggested_chunk_size = (
                    smallest_document_token_count
                    + (self.chunk_overlap * (num_context_per_document - 1))
                ) // num_context_per_document
                raise ValueError(
                    f"Your smallest document is only sized {smallest_document_token_count} tokens."
                    f"Please adjust the chunk_size to no more than {suggested_chunk_size}."
                )

        # Chunk docs if not already cached via ChromaDB
        if self.source_files_to_collections_map == None:
            self.source_files_to_collections_map = {}
        if self.doc_to_chunker_map != None:
            for key, chunker in tqdm_bar(
                self.doc_to_chunker_map.items(), "✨ 📚 ✨ Chunking Documents"
            ):
                self.source_files_to_collections_map[key] = chunker.chunk_doc(
                    self.chunk_size, self.chunk_overlap
                )

        # Progress Bar
        p_bar = tqdm_bar(
            total=3
            * sum(
                min(num_context_per_document, collection.count())
                for _, collection in self.source_files_to_collections_map.items()
            ),
            desc="✨ 🧩 ✨ Generating Contexts",
        )

        # Generate contexts
        self.total_chunks = 0
        for path, collection in self.source_files_to_collections_map.items():
            num_chunks = collection.count()
            min_num_context = min(num_context_per_document, num_chunks)
            contexts_per_doc, scores_per_doc = (
                self._get_n_random_contexts_per_doc(
                    path=path,
                    n_contexts_per_doc=min_num_context,
                    context_size=max_context_size,
                    similarity_threshold=self.similarity_threshold,
                    p_bar=p_bar,
                )
            )
            contexts.extend(contexts_per_doc)
            scores.extend(scores_per_doc)
            for _ in contexts_per_doc:
                source_files.append(path)
            self.total_chunks += num_chunks
        return contexts, source_files, scores

    async def a_generate_contexts(
        self, num_context_per_document: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        self.check_if_docs_are_loaded()
        scores = []
        contexts = []
        source_files = []

        # Check if chunk_size is valid for document lengths
        if self.doc_to_chunker_map is not None:
            smallest_document_token_count = min(
                chunker.text_token_count
                for chunker in self.doc_to_chunker_map.values()
            )
            smallest_document_num_chunks = 1 + math.floor(
                (smallest_document_token_count - self.chunk_size)
                / (self.chunk_size - self.chunk_overlap)
            )
            if smallest_document_num_chunks < num_context_per_document:
                suggested_chunk_size = (
                    smallest_document_token_count
                    + (self.chunk_overlap * (num_context_per_document - 1))
                ) // num_context_per_document
                raise ValueError(
                    f"Your smallest document is only sized {smallest_document_token_count} tokens."
                    f"Please adjust the chunk_size to no more than {suggested_chunk_size}."
                )

        # Chunk docs if not already cached via ChromaDB
        async def a_chunk_and_store(key, chunker: DocumentChunker):
            self.source_files_to_collections_map[key] = (
                await chunker.a_chunk_doc(self.chunk_size, self.chunk_overlap)
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

        # Progress Bar
        p_bar = tqdm_bar(
            total=3
            * sum(
                min(num_context_per_document, collection.count())
                for _, collection in self.source_files_to_collections_map.items()
            ),
            desc="✨ 🧩 ✨ Generating Contexts",
        )

        # Generate contexts
        self.total_chunks = 0
        tasks = [
            self._a_process_document_async(
                path,
                collection,
                num_context_per_document,
                max_context_size,
                p_bar,
            )
            for path, collection in self.source_files_to_collections_map.items()
        ]
        results = await asyncio.gather(*tasks)
        for path, contexts_per_doc, scores_per_doc, num_chunks in results:
            contexts.extend(contexts_per_doc)
            scores.extend(scores_per_doc)
            for _ in contexts_per_doc:
                source_files.append(path)
            self.total_chunks += num_chunks

        return contexts, source_files, scores

    async def _a_process_document_async(
        self,
        path: str,
        collection,
        num_context_per_document: int,
        max_context_size: int,
        p_bar: tqdm_bar,
    ):
        num_chunks = collection.count()
        min_num_context = min(num_context_per_document, num_chunks)
        contexts_per_doc, scores_per_doc = (
            await self._a_get_n_random_contexts_per_doc(
                path=path,
                n_contexts_per_doc=min_num_context,
                context_size=max_context_size,
                similarity_threshold=self.similarity_threshold,
                p_bar=p_bar,
            )
        )
        return path, contexts_per_doc, scores_per_doc, num_chunks

    #########################################################
    ### Generate Contexts from Nodes ########################
    #########################################################

    def generate_contexts_from_nodes(
        self, num_context: int, max_chunks_per_context: int = 3
    ):
        scores = []
        contexts = []
        source_files = []

        # Determine the number of contexts to generate
        nodes_collection = self.source_files_to_collections_map["nodes"]
        num_chunks = nodes_collection.count()
        num_context = min(num_context, num_chunks)

        # Progress Bar
        p_bar = tqdm_bar(
            total=max_chunks_per_context * num_context,
            desc="✨ 🧩 ✨ Generating Contexts",
        )

        # Generate contexts
        self.total_chunks = 0
        for path, _ in self.source_files_to_collections_map.items():
            contexts_per_doc, scores_per_doc = (
                self._get_n_random_contexts_per_doc(
                    path=path,
                    n_contexts_per_doc=num_context,
                    context_size=max_chunks_per_context,
                    similarity_threshold=self.similarity_threshold,
                    p_bar=p_bar,
                )
            )
            contexts.extend(contexts_per_doc)
            scores.extend(scores_per_doc)
            for _ in contexts_per_doc:
                source_files.append(path)
            self.total_chunks += num_chunks

        return contexts, source_files, scores

    #########################################################
    ### Get Random Contexts #################################
    #########################################################

    def _get_n_random_contexts_per_doc(
        self,
        path: str,
        n_contexts_per_doc: int,
        context_size: int,
        similarity_threshold: int,
        p_bar: tqdm_bar,
    ):
        assert (
            n_contexts_per_doc > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."
        contexts = []
        scores = []
        num_query_docs = 0

        # Sample random chunks
        random_chunks, scores = self._get_n_random_chunks_per_doc(
            path=path, n_chunks=n_contexts_per_doc, p_bar=p_bar
        )
        collection = self.source_files_to_collections_map[path]

        # Find similar chunks for sampled random chunks
        for i in range(len(random_chunks)):
            random_chunk = random_chunks[i]
            context = [random_chunk]

            # Disregard empty chunks
            if not random_chunk.strip():
                continue

            # Query for similar chunks
            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk),
                n_results=min(context_size, collection.count()),
            )

            # disregard repeated chunks and chunks that don't pass the similarity threshold
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):
                similar_chunk_similarity = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity > similarity_threshold
                ):
                    context.append(similar_chunk_text)
            contexts.append(context)

        return contexts, scores

    async def _a_get_n_random_contexts_per_doc(
        self,
        path: str,
        n_contexts_per_doc: int,
        context_size: int,
        similarity_threshold: int,
        p_bar: tqdm_bar,
    ):
        assert (
            n_contexts_per_doc > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."
        contexts = []
        scores = []
        num_query_docs = 0

        # Sample random chunks
        random_chunks, scores = await self._a_get_n_random_chunks_per_doc(
            path=path, n_chunks=n_contexts_per_doc, p_bar=p_bar
        )
        collection = self.source_files_to_collections_map[path]

        # Find similar chunks for sampled random chunks
        for i in range(len(random_chunks)):
            random_chunk = random_chunks[i]
            context = [random_chunk]

            # Disregard empty chunks
            if not random_chunk.strip():
                continue

            # Query for similar chunks
            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk),
                n_results=min(context_size, collection.count()),
            )

            # disregard repeated chunks and chunks that don't pass the similarity threshold
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):
                similar_chunk_similarity = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity > similarity_threshold
                ):
                    context.append(similar_chunk_text)
            contexts.append(context)

        return contexts, scores

    #########################################################
    ### Get Random Chunks ###################################
    #########################################################

    def _get_n_random_chunks_per_doc(
        self, path: str, n_chunks: int, p_bar: tqdm_bar
    ) -> Tuple[List[str], List[float]]:

        # Determine the number of chunks to sample
        collection = self.source_files_to_collections_map[path]
        total_chunks = collection.count()
        assert (
            n_chunks <= total_chunks
        ), f"Requested {n_chunks} chunks, but the collection only contains {total_chunks} chunks."
        if total_chunks >= n_chunks * self.max_retries:
            sample_size = n_chunks * self.max_retries
        else:
            sample_size = n_chunks

        # Randomly sample chunks from collection
        random_ids = [
            str(i) for i in random.sample(range(total_chunks), sample_size)
        ]
        chunks = collection.get(ids=random_ids)["documents"]
        if total_chunks < n_chunks * self.max_retries:
            scores = []
            for chunk in chunks:
                score = self.evaluate_chunk(chunk)
                scores.append(score)
                p_bar.update(3)
            return chunks, scores

        # Evaluate sampled chunks
        evaluated_chunks = []
        scores = []
        retry_count = 0
        for i, chunk in enumerate(chunks):
            score = self.evaluate_chunk(chunk)
            if score > self.filter_threshold:
                p_bar.update(self.max_retries - retry_count)
                evaluated_chunks.append(chunk)
                scores.append(score)
                retry_count = 0
            else:
                p_bar.update(1)
                retry_count += 1
                if retry_count == self.max_retries:
                    evaluated_chunks.append(chunk)
                    scores.append(score)
                    retry_count = 0
            if len(evaluated_chunks) == n_chunks:
                break

        return evaluated_chunks, scores

    async def _a_get_n_random_chunks_per_doc(
        self, path: str, n_chunks: int, p_bar: tqdm_bar
    ) -> Tuple[List[str], List[float]]:

        # Determine the number of chunks to sample
        collection = self.source_files_to_collections_map[path]
        total_chunks = collection.count()
        assert (
            n_chunks <= total_chunks
        ), f"Requested {n_chunks} chunks, but the collection only contains {total_chunks} chunks."
        if total_chunks >= n_chunks * self.max_retries:
            sample_size = n_chunks * self.max_retries
        else:
            sample_size = n_chunks

        # Randomly sample chunks from collection
        random_ids = [
            str(i) for i in random.sample(range(total_chunks), sample_size)
        ]
        chunks = collection.get(ids=random_ids)["documents"]
        if total_chunks < n_chunks * self.max_retries:
            return chunks, [
                self.evaluate_chunk(chunk)
                for chunk in chunks
                if not p_bar.update(3)
            ]

        # Evaluate sampled chunks
        async def a_evaluate_chunk_and_update(chunk):
            score = await self.a_evaluate_chunk(chunk)
            p_bar.update(1)
            return score

        tasks = [a_evaluate_chunk_and_update(chunk) for chunk in chunks]
        scores = await asyncio.gather(*tasks)
        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        best_chunks = [pair[0] for pair in chunk_score_pairs[:n_chunks]]
        best_scores = [pair[1] for pair in chunk_score_pairs[:n_chunks]]

        return best_chunks, best_scores

    #########################################################
    ### Evaluate Chunk Quality ##############################
    #########################################################

    def evaluate_chunk(self, chunk) -> float:
        prompt = FilterTemplate.evaluate_context(chunk)
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ContextScore)
            self.total_cost += cost
            return (res.clarity + res.depth + res.structure + res.relevance) / 4
        else:
            try:
                res: ContextScore = self.model.generate(
                    prompt, schema=ContextScore
                )
                return (
                    res.clarity + res.depth + res.structure + res.relevance
                ) / 4
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                score = (
                    data["clarity"]
                    + data["depth"]
                    + data["structure"]
                    + data["relevance"]
                ) / 4
                return score

    async def a_evaluate_chunk(self, chunk) -> float:
        prompt = FilterTemplate.evaluate_context(chunk)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=ContextScore)
            self.total_cost += cost
            return (res.clarity + res.depth + res.structure + res.relevance) / 4
        else:

            try:
                res: ContextScore = await self.model.a_generate(
                    prompt, schema=ContextScore
                )
                return (
                    res.clarity + res.depth + res.structure + res.relevance
                ) / 4
            except TypeError:
                res: ContextScore = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                score = (
                    data["clarity"]
                    + data["depth"]
                    + data["structure"]
                    + data["relevance"]
                ) / 4
                return score

    #########################################################
    ### Load Docs ###########################################
    #########################################################

    def _load_docs(self):
        import chromadb

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
                doc_chunker = DocumentChunker(self.embedder)
                doc_chunker.load_doc(path)
                if path not in self.doc_to_chunker_map:
                    self.doc_to_chunker_map[path] = doc_chunker

    async def _a_load_docs(self):
        import chromadb

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

            except:
                if self.doc_to_chunker_map is None:
                    self.doc_to_chunker_map = {}
                doc_chunker = DocumentChunker(self.embedder)
                await doc_chunker.a_load_doc(path)
                if path not in self.doc_to_chunker_map:
                    self.doc_to_chunker_map[path] = doc_chunker

        # Process all documents asynchronously with a progress bar
        tasks = [a_process_document(path) for path in self.document_paths]
        await tqdm_asyncio.gather(*tasks, desc="✨ 🚀 ✨ Loading Documents")

    #########################################################
    ### Load Nodes ##########################################
    #########################################################

    def _load_nodes(self):
        import chromadb
        from chromadb.errors import InvalidCollectionException

        for _ in tqdm_bar(range(1), "✨ 🚀 ✨ Loading Nodes"):
            try:
                # Create ChromaDB client
                client = chromadb.PersistentClient(
                    path=f".vector_db/{self._nodes[0].id_}"
                )
                collection = client.get_collection(name=f"processed_chunks")
                self.source_files_to_collections_map = {}
                self.source_files_to_collections_map["nodes"] = collection

            except InvalidCollectionException:
                doc_chunker = DocumentChunker(self.embedder)
                collection = doc_chunker.from_nodes(self._nodes)
                self.source_files_to_collections_map = {}
                self.source_files_to_collections_map["nodes"] = collection

    async def _a_load_nodes(self):
        import chromadb

        async def a_process_nodes(path):
            try:
                # Create ChromaDB client
                client = chromadb.PersistentClient(
                    path=f".vector_db/{self._nodes[0].id_}"
                )
                collection = client.get_collection(name=f"processed_chunks")
                # Needs to strictly be after getting collection so map is assigned to None if exception is raised
                if self.source_files_to_collections_map == None:
                    self.source_files_to_collections_map = {}
                self.source_files_to_collections_map["nodes"] = collection

            except:
                if self.doc_to_chunker_map == None:
                    self.doc_to_chunker_map = {}
                doc_chunker = DocumentChunker(self.embedder)
                collection = await doc_chunker.a_from_nodes(self._nodes)
                self.source_files_to_collections_map["nodes"] = collection

        # Process all documents asynchronously with a progress bar
        tasks = [a_process_nodes() for _ in range(1)]
        await tqdm_asyncio.gather(*tasks, desc="✨ 🚀 ✨ Loading Nodes")

    #########################################################
    ### Check Docs Loaded ###################################
    #########################################################

    def check_if_docs_are_loaded(self):
        if (
            self.doc_to_chunker_map == None
            and self.source_files_to_collections_map == None
        ):
            raise ValueError(
                "Context Generator has yet to properly load documents"
            )
