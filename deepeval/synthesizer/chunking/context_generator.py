import shutil
from typing import List, Tuple, Dict, Optional, Union
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
        encoding: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        max_retries: int = 3,
        filter_threshold: float = 0.5,
        similarity_threshold: float = 0.5,
    ):
        # Ensure document_paths is provided
        if not document_paths:
            raise ValueError("`document_path` is empty or missing.")
        if chunk_overlap > chunk_size - 1:
            raise ValueError(
                f"`chunk_overlap` must not exceed {chunk_size - 1} (chunk_size - 1)."
            )

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.total_chunks = 0
        self.document_paths: List[str] = document_paths
        self.encoding = encoding

        # Model parameters
        self.model, self.using_native_model = initialize_model(model)
        self.embedder = embedder

        # Quality parameters
        self.max_retries = max_retries
        self.filter_threshold = filter_threshold
        self.similarity_threshold = similarity_threshold
        self.not_enough_chunks = False

        # cost tracking
        self.total_cost = 0.0

    #########################################################
    ### Generate Contexts ###################################
    #########################################################

    def generate_contexts(
        self,
        max_contexts_per_source_file: int,
        min_contexts_per_source_file: int,
        max_context_size: int = 3,
        min_context_size: int = 1,
    ) -> Tuple[List[List[str]], List[str], List[float]]:
        from chromadb.api.models.Collection import Collection

        vector_db_path = ".vector_db"
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)

        try:
            # Initialize lists for scores, contexts, and source files
            scores = []
            contexts = []
            source_files = []

            # Load the source files and create document chunkers for each file
            source_file_to_chunker_map: Dict[str, DocumentChunker] = (
                self._load_docs()
            )

            # Chunk each file into a chroma collection of chunks
            source_files_to_chunk_collections_map: Dict[str, Collection] = {}
            for key, chunker in tqdm_bar(
                source_file_to_chunker_map.items(),
                "✨ 📚 ✨ Chunking Documents",
            ):
                collection = chunker.chunk_doc(
                    self.chunk_size, self.chunk_overlap
                )
                self.validate_chunk_size(
                    min_contexts_per_source_file, collection
                )
                source_files_to_chunk_collections_map[key] = collection

            # Initialize progress bar for context generation
            num_contexts = sum(
                min(max_contexts_per_source_file, collection.count())
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            self.total_chunks = sum(
                collection.count()
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            generation_p_bar = tqdm_bar(
                total=self.max_retries * num_contexts,
                desc="✨ 🧩 ✨ Generating Contexts",
                leave=True,
            )

            # Generate contexts for each source file
            for (
                path,
                collection,
            ) in source_files_to_chunk_collections_map.items():
                self.validate_context_size(min_context_size, path, collection)
                max_context_size = min(max_context_size, collection.count())
                contexts_per_source_file, scores_per_source_file = (
                    self._generate_contexts_per_source_file(
                        path=path,
                        n_contexts_per_source_file=min(
                            max_contexts_per_source_file, collection.count()
                        ),
                        context_size=max_context_size,
                        similarity_threshold=self.similarity_threshold,
                        generation_p_bar=generation_p_bar,
                        source_files_to_collections_map=source_files_to_chunk_collections_map,
                    )
                )
                contexts.extend(contexts_per_source_file)
                scores.extend(scores_per_source_file)
                source_files.extend([path] * len(contexts_per_source_file))
            generation_p_bar.close()

            if self.not_enough_chunks:
                print(
                    f"Not enough available chunks in smallest document to evaluate chunk quality using the filter threshold: {self.filter_threshold}."
                )

            return contexts, source_files, scores

        finally:
            # Always delete the .vector_db folder if it exists, regardless of success or failure
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)

    async def a_generate_contexts(
        self,
        max_contexts_per_source_file: int,
        min_contexts_per_source_file: int,
        max_context_size: int = 3,
        min_context_size: int = 1,
    ) -> Tuple[List[List[str]], List[str], List[float]]:
        from chromadb.api.models.Collection import Collection

        vector_db_path = ".vector_db"
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)

        try:
            # Initialize lists for scores, contexts, and source files
            scores = []
            contexts = []
            source_files = []

            # Check if chunk_size and max_context_size is valid for document lengths
            source_file_to_chunker_map: Dict[str, DocumentChunker] = (
                await self._a_load_docs()
            )

            # Chunk each file into a chroma collection of chunks
            source_files_to_chunk_collections_map: Dict[str, Collection] = {}

            async def a_chunk_and_store(key, chunker: DocumentChunker):
                collection = await chunker.a_chunk_doc(
                    self.chunk_size, self.chunk_overlap
                )
                self.validate_chunk_size(
                    min_contexts_per_source_file, collection
                )
                source_files_to_chunk_collections_map[key] = collection

            tasks = [
                a_chunk_and_store(key, chunker)
                for key, chunker in source_file_to_chunker_map.items()
            ]
            await tqdm_asyncio.gather(
                *tasks, desc="✨ 📚 ✨ Chunking Documents"
            )

            # Initialize progress bar for context generation
            num_contexts = sum(
                min(max_contexts_per_source_file, collection.count())
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            self.total_chunks = sum(
                collection.count()
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            generation_p_bar = tqdm_bar(
                total=self.max_retries * num_contexts,
                desc="✨ 🧩 ✨ Generating Contexts",
                leave=True,
            )

            # Generate contexts for each source file
            tasks = []
            for (
                path,
                collection,
            ) in source_files_to_chunk_collections_map.items():
                self.validate_context_size(min_context_size, path, collection)
                max_context_size = min(max_context_size, collection.count())
                tasks.append(
                    self._a_process_document_async(
                        path,
                        min(max_contexts_per_source_file, collection.count()),
                        max_context_size,
                        generation_p_bar,
                        source_files_to_chunk_collections_map,
                    )
                )

            results = await asyncio.gather(*tasks)
            for path, contexts_per_doc, scores_per_doc in results:
                contexts.extend(contexts_per_doc)
                scores.extend(scores_per_doc)
                for _ in contexts_per_doc:
                    source_files.append(path)
            generation_p_bar.close()

            if self.not_enough_chunks:
                print(
                    f"Not enough available chunks in smallest document to evaluate chunk quality using the filter threshold: {self.filter_threshold}."
                )

            return contexts, source_files, scores

        finally:
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)

    async def _a_process_document_async(
        self,
        path: str,
        num_context_per_source_file: int,
        max_context_size: int,
        generation_p_bar: tqdm_bar,
        source_files_to_collections_map: Dict,
    ):
        contexts_per_doc, scores_per_doc = (
            await self._a_get_n_random_contexts_per_source_file(
                path=path,
                n_contexts_per_source_file=num_context_per_source_file,
                context_size=max_context_size,
                similarity_threshold=self.similarity_threshold,
                generation_p_bar=generation_p_bar,
                source_files_to_collections_map=source_files_to_collections_map,
            )
        )
        return path, contexts_per_doc, scores_per_doc

    #########################################################
    ### Get Generate Contexts for Each Source File ##########
    #########################################################

    def _generate_contexts_per_source_file(
        self,
        path: str,
        n_contexts_per_source_file: int,
        context_size: int,
        similarity_threshold: float,
        generation_p_bar: tqdm_bar,
        source_files_to_collections_map: Dict,
    ):
        assert (
            n_contexts_per_source_file > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."

        # Initialize lists for scores, contexts
        contexts = []
        scores = []
        num_query_docs = 0
        collection = source_files_to_collections_map[path]

        # Sample n random chunks from each doc (each random chunk is the first chunk in each context)
        filling_p_bar = tqdm_bar(
            total=(context_size - 1) * n_contexts_per_source_file,
            desc="  ✨ 🫗 ✨ Filling Contexts",
            leave=False,
        )
        random_chunks, scores = self._get_n_random_chunks_per_source_file(
            path,
            n_contexts_per_source_file,
            generation_p_bar,
            source_files_to_collections_map,
        )

        # Find similar chunks for each context
        for random_chunk in random_chunks:
            # Create context
            context = [random_chunk]

            # Disregard empty chunks
            if not random_chunk.strip():
                filling_p_bar.update(context_size - 1)
                continue

            # Query for similar chunks
            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk), n_results=context_size
            )

            # Disregard repeated chunks and chunks that don't pass the similarity threshold
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):

                # Calculate chunk similarity score
                similar_chunk_similarity_score = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity_score > similarity_threshold
                ):
                    context.append(similar_chunk_text)
                if j != 0:
                    filling_p_bar.update(1)

            contexts.append(context)

        return contexts, scores

    async def _a_get_n_random_contexts_per_source_file(
        self,
        path: str,
        n_contexts_per_source_file: int,
        context_size: int,
        similarity_threshold: float,
        generation_p_bar: tqdm_bar,
        source_files_to_collections_map: Dict,
    ):
        assert (
            n_contexts_per_source_file > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."

        # Initialize lists for scores, contexts
        contexts = []
        scores = []
        num_query_docs = 0
        collection = source_files_to_collections_map[path]

        # Sample n random chunks from each doc (each random chunk is the first chunk in each context)
        filling_p_bar = tqdm_bar(
            total=(context_size - 1) * n_contexts_per_source_file,
            desc="  ✨ 🫗 ✨ Filling Contexts",
            leave=False,
        )
        random_chunks, scores = (
            await self._a_get_n_random_chunks_per_source_file(
                path,
                n_contexts_per_source_file,
                generation_p_bar,
                source_files_to_collections_map,
            )
        )

        # Find similar chunks for each context
        for random_chunk in random_chunks:
            # Create context
            context = [random_chunk]

            # Disregard empty chunks
            if not random_chunk.strip():
                filling_p_bar.update(context_size - 1)
                continue

            # Query for similar chunks
            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk), n_results=context_size
            )

            # Disregard repeated chunks and chunks that don't pass the similarity threshold
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):

                # Calculate chunk similarity score
                similar_chunk_similarity_score = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity_score > similarity_threshold
                ):
                    context.append(similar_chunk_text)
                if j != 0:
                    filling_p_bar.update(1)

            contexts.append(context)

        return contexts, scores

    #########################################################
    ### Get Random Chunks ###################################
    #########################################################

    def _get_n_random_chunks_per_source_file(
        self,
        path: str,
        n_chunks: int,
        p_bar: tqdm_bar,
        source_files_to_collections_map: Dict,
    ) -> Tuple[List[str], List[float]]:
        collection = source_files_to_collections_map[path]
        total_chunks = collection.count()

        # Determine sample size:
        if total_chunks >= n_chunks * self.max_retries:
            sample_size = n_chunks * self.max_retries
        else:
            sample_size = n_chunks

        # Randomly sample chunks
        random_ids = [
            str(i) for i in random.sample(range(total_chunks), sample_size)
        ]
        chunks = collection.get(ids=random_ids)["documents"]

        # If total_chunks is less than n_chunks * max_retries, simply evaluate all chunks
        if total_chunks < n_chunks * self.max_retries:
            self.not_enough_chunks = True
            scores = []
            for chunk in chunks:
                score = self.evaluate_chunk(chunk)
                scores.append(score)
                p_bar.update(self.max_retries)
            return chunks, scores

        # Evaluate sampled chunks
        evaluated_chunks = []
        scores = []
        retry_count = 0
        for chunk in chunks:
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

    async def _a_get_n_random_chunks_per_source_file(
        self,
        path: str,
        n_chunks: int,
        p_bar: tqdm_bar,
        source_files_to_collections_map: Dict,
    ) -> Tuple[List[str], List[float]]:
        collection = source_files_to_collections_map[path]
        total_chunks = collection.count()

        # Determine sample size:
        if total_chunks >= n_chunks * self.max_retries:
            sample_size = n_chunks * self.max_retries
        else:
            sample_size = n_chunks

        # Randomly sample chunks
        random_ids = [
            str(i) for i in random.sample(range(total_chunks), sample_size)
        ]
        chunks = collection.get(ids=random_ids)["documents"]

        # If total_chunks is less than n_chunks * max_retries, simply evaluate all chunks
        if total_chunks < n_chunks * self.max_retries:
            self.not_enough_chunks = True

            async def update_and_evaluate(chunk):
                p_bar.update(self.max_retries)
                return await self.a_evaluate_chunk(chunk)

            scores = await asyncio.gather(
                *(update_and_evaluate(chunk) for chunk in chunks)
            )
            return chunks, scores

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
    ### Validation ##########################################
    #########################################################

    def validate_context_size(
        self,
        min_context_size: int,
        path: str,
        collection,
    ):
        collection_size = collection.count()
        if collection_size < min_context_size:
            error_message = [
                f"{path} has {collection_size} chunks, which is less than the minimum context size of {min_context_size}",
                f"Adjust the `min_context_length` to no more than {collection_size}.",
            ]
            raise ValueError("\n".join(error_message))

    def validate_chunk_size(
        self,
        min_contexts_per_source_file: int,
        collection,
    ):
        # Calculate the number of chunks the smallest document can produce.
        document_token_count = collection.count()
        document_num_chunks = 1 + math.floor(
            max(document_token_count - self.chunk_size, 0)
            / (self.chunk_size - self.chunk_overlap)
        )

        # If not enough chunks are produced, raise an error with suggestions.
        if document_num_chunks < min_contexts_per_source_file:

            # Build the error message with suggestions.
            error_lines = [
                f"Impossible to generate {min_contexts_per_source_file} contexts from a document of size {document_token_count}.",
                "You have the following options:",
            ]
            suggestion_num = 1

            # 1. Suggest adjusting the number of contexts if applicable.
            if document_num_chunks > 0:
                error_lines.append(
                    f"{suggestion_num}. Adjust the `min_contexts_per_document` to no more than {document_num_chunks}."
                )
                suggestion_num += 1

            # 2. Determine whether to suggest adjustments for chunk_size.
            suggested_chunk_size = (
                document_token_count
                + (self.chunk_overlap * (min_contexts_per_source_file - 1))
            ) // min_contexts_per_source_file
            adjust_chunk_size = (
                suggested_chunk_size > 0
                and suggested_chunk_size > self.chunk_overlap
            )
            if adjust_chunk_size:
                error_lines.append(
                    f"{suggestion_num}. Adjust the `chunk_size` to no more than {suggested_chunk_size}."
                )
                suggestion_num += 1

            # 3. Determine whether to suggest adjustments for chunk_overlap.
            if min_contexts_per_source_file > 1:
                suggested_overlap = (
                    (
                        (min_contexts_per_source_file * self.chunk_size)
                        - document_num_chunks
                    )
                    // (min_contexts_per_source_file - 1)
                ) + 1
                adjust_overlap = (
                    suggested_overlap > 0
                    and self.chunk_size > suggested_overlap
                )
                if adjust_overlap:
                    error_lines.append(
                        f"{suggestion_num}. Adjust the `chunk_overlap` to at least {suggested_overlap}."
                    )
                    suggestion_num += 1

            # 4. If either individual adjustment is suggested, also offer a combined adjustment option.
            if adjust_chunk_size or adjust_overlap:
                error_lines.append(
                    f"{suggestion_num}. Adjust both the `chunk_size` and `chunk_overlap`."
                )
            error_message = "\n".join(error_lines)
            raise ValueError(error_message)

    #########################################################
    ### Loading documents and chunkers ######################
    #########################################################

    def _load_docs(self):
        doc_to_chunker_map = {}
        for path in tqdm_bar(self.document_paths, "✨ 🚀 ✨ Loading Documents"):
            doc_chunker = DocumentChunker(self.embedder)
            doc_chunker.load_doc(path, self.encoding)
            doc_to_chunker_map[path] = doc_chunker
        return doc_to_chunker_map

    async def _a_load_docs(self):
        doc_to_chunker_map = {}

        async def a_process_document(path):
            doc_chunker = DocumentChunker(self.embedder)
            await doc_chunker.a_load_doc(path, self.encoding)
            doc_to_chunker_map[path] = doc_chunker

        tasks = [a_process_document(path) for path in self.document_paths]
        await tqdm_asyncio.gather(*tasks, desc="✨ 🚀 ✨ Loading Documents")
        return doc_to_chunker_map
