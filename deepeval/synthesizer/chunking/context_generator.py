from typing import List, Tuple, Dict, Optional, Union
from rich.progress import Progress
from pydantic import BaseModel
import asyncio
import shutil
import random
import atexit
import time
import math
import sys
import os
import gc


from deepeval.synthesizer.utils import (
    print_synthesizer_status,
    SynthesizerStatus,
)
from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.synthesizer.templates.template import FilterTemplate
from deepeval.models.base_model import (
    DeepEvalBaseEmbeddingModel,
    DeepEvalBaseLLM,
)
from deepeval.utils import update_pbar, add_pbar, remove_pbars

# Monkey patch shutil.rmtree to handle locked files better on Windows
original_rmtree = shutil.rmtree


def safe_rmtree(
    path,
    *args,
    **kwargs,
):
    if not os.path.exists(path):
        return
    for _ in range(3):
        try:
            gc.collect()
            time.sleep(1)
            if sys.platform == "win32":
                os.system(f'attrib -r -s -h "{path}\\*" /s /d')
            kwargs["ignore_errors"] = True
            original_rmtree(path, *args, **kwargs)
            print_synthesizer_status(
                SynthesizerStatus.SUCCESS,
                "Successfully deleted",
                path,
            )
            return
        except Exception as e:
            print_synthesizer_status(
                SynthesizerStatus.WARNING,
                "Delete attempt failed",
                f"{e}",
            )
            time.sleep(2)
    print_synthesizer_status(
        SynthesizerStatus.FAILURE,
        "Unable to delete",
        path,
    )


def close_chroma_clients():
    gc.collect()
    time.sleep(1)


atexit.register(close_chroma_clients)
shutil.rmtree = safe_rmtree


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

        # cost and progress tracking
        self.total_cost = 0.0
        self.context_number = 0
        self.pbar_filling_contexts_ids = []

    #########################################################
    ### Generate Contexts ###################################
    #########################################################

    def generate_contexts(
        self,
        max_contexts_per_source_file: int,
        min_contexts_per_source_file: int,
        max_context_size: int = 3,
        min_context_size: int = 1,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
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

            pbar_load_docs_id = add_pbar(
                progress,
                f"\t📚 Loading {len(self.document_paths)} documents",
                len(self.document_paths),
            )
            pbar_chunk_docs_id = add_pbar(
                progress,
                f"\t🍫 Chunking {len(self.document_paths)} documents",
                len(self.document_paths),
            )
            pbar_generate_contexts_id = add_pbar(
                progress,
                f"\t🚧 Constructing up to {len(self.document_paths) * max_contexts_per_source_file} contexts",
                1,
            )
            self.pbar_load_docs_id = pbar_load_docs_id
            self.pbar_chunk_docs_id = pbar_chunk_docs_id
            self.pbar_generate_contexts_id = pbar_generate_contexts_id

            # Load the source files and create document chunkers for each file
            source_file_to_chunker_map: Dict[str, DocumentChunker] = (
                self._load_docs(progress, pbar_load_docs_id)
            )
            update_pbar(progress, pbar_id, remove=False)

            # Chunk each file into a chroma collection of chunks
            source_files_to_chunk_collections_map: Dict[str, Collection] = {}
            for key, chunker in source_file_to_chunker_map.items():
                collection = chunker.chunk_doc(
                    self.chunk_size, self.chunk_overlap
                )
                self.validate_chunk_size(
                    min_contexts_per_source_file, collection
                )
                source_files_to_chunk_collections_map[key] = collection
                update_pbar(progress, pbar_chunk_docs_id, remove=False)
            update_pbar(progress, pbar_id, remove=False)

            # Initialize progress bar for context generation
            num_contexts = sum(
                min(max_contexts_per_source_file, collection.count())
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            self.total_chunks = sum(
                collection.count()
                for _, collection in source_files_to_chunk_collections_map.items()
            )

            # Update progress bar total length for context generation after determining number of contexts
            progress.update(
                pbar_generate_contexts_id,
                total=(self.max_retries + max_context_size - 1) * num_contexts,
                completed=0,
            )

            # Generate contexts for each source file
            for (
                path,
                collection,
            ) in source_files_to_chunk_collections_map.items():
                self.validate_context_size(min_context_size, path, collection)
                max_context_size = min(max_context_size, collection.count())
                num_context_per_source_file = min(
                    max_contexts_per_source_file, collection.count()
                )
                contexts_per_source_file, scores_per_source_file = (
                    self._generate_contexts_per_source_file(
                        path=path,
                        n_contexts_per_source_file=num_context_per_source_file,
                        context_size=max_context_size,
                        similarity_threshold=self.similarity_threshold,
                        source_files_to_collections_map=source_files_to_chunk_collections_map,
                        progress=progress,
                        pbar_generate_contexts_id=pbar_generate_contexts_id,
                    )
                )
                contexts.extend(contexts_per_source_file)
                scores.extend(scores_per_source_file)
                source_files.extend([path] * len(contexts_per_source_file))
            update_pbar(
                progress,
                pbar_generate_contexts_id,
                advance_to_end=True,
                remove=False,
            )
            update_pbar(progress, pbar_id, remove=False)
            remove_pbars(progress, self.pbar_filling_contexts_ids)

            if self.not_enough_chunks:
                print_synthesizer_status(
                    SynthesizerStatus.WARNING,
                    "Filtering not applied",
                    f"Nnot enough chunks in smallest document",
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
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
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
            pbar_load_docs_id = add_pbar(
                progress,
                f"\t📚 Loading {len(self.document_paths)} documents",
                len(self.document_paths),
            )
            pbar_chunk_docs_id = add_pbar(
                progress,
                f"\t🍫 Chunking {len(self.document_paths)} documents",
                len(self.document_paths),
            )
            pbar_generate_contexts_id = add_pbar(
                progress,
                f"\t🚧 Constructing up to {len(self.document_paths) * max_contexts_per_source_file} contexts",
                1,
            )
            self.pbar_load_docs_id = pbar_load_docs_id
            self.pbar_chunk_docs_id = pbar_chunk_docs_id
            self.pbar_generate_contexts_id = pbar_generate_contexts_id

            source_file_to_chunker_map: Dict[str, DocumentChunker] = (
                await self._a_load_docs(progress, pbar_load_docs_id)
            )
            update_pbar(progress, pbar_id, remove=False)

            # Chunk each file into a chroma collection of chunks
            async def a_chunk_and_store(
                key,
                chunker: DocumentChunker,
                progress: Optional[Progress] = None,
                pbar_chunk_docs_id: Optional[int] = None,
            ):
                collection = await chunker.a_chunk_doc(
                    self.chunk_size, self.chunk_overlap
                )
                self.validate_chunk_size(
                    min_contexts_per_source_file, collection
                )
                source_files_to_chunk_collections_map[key] = collection
                update_pbar(progress, pbar_chunk_docs_id, remove=False)

            source_files_to_chunk_collections_map: Dict[str, Collection] = {}
            tasks = [
                a_chunk_and_store(key, chunker, progress, pbar_chunk_docs_id)
                for key, chunker in source_file_to_chunker_map.items()
            ]
            await asyncio.gather(*tasks)
            update_pbar(progress, pbar_id, remove=False)

            # Initialize progress bar for context generation
            num_contexts = sum(
                min(max_contexts_per_source_file, collection.count())
                for _, collection in source_files_to_chunk_collections_map.items()
            )
            self.total_chunks = sum(
                collection.count()
                for _, collection in source_files_to_chunk_collections_map.items()
            )

            # Update progress bar total length for context generation after determining number of contexts
            progress.update(
                pbar_generate_contexts_id,
                total=(self.max_retries + max_context_size - 1) * num_contexts,
                completed=0,
            )

            # Generate contexts for each source file
            tasks = []
            for (
                path,
                collection,
            ) in source_files_to_chunk_collections_map.items():
                self.validate_context_size(min_context_size, path, collection)
                max_context_size = min(max_context_size, collection.count())
                n_contexts_per_source_file = min(
                    max_contexts_per_source_file, collection.count()
                )
                tasks.append(
                    self._a_process_document_async(
                        path=path,
                        num_context_per_source_file=n_contexts_per_source_file,
                        max_context_size=max_context_size,
                        source_files_to_collections_map=source_files_to_chunk_collections_map,
                        progress=progress,
                        pbar_generate_contexts_id=pbar_generate_contexts_id,
                    )
                )

            results = await asyncio.gather(*tasks)
            update_pbar(
                progress,
                pbar_generate_contexts_id,
                advance_to_end=True,
                remove=False,
            )
            update_pbar(progress, pbar_id, remove=False)
            remove_pbars(progress, self.pbar_filling_contexts_ids)
            for path, contexts_per_doc, scores_per_doc in results:
                contexts.extend(contexts_per_doc)
                scores.extend(scores_per_doc)
                for _ in contexts_per_doc:
                    source_files.append(path)

            if self.not_enough_chunks:
                print_synthesizer_status(
                    SynthesizerStatus.WARNING,
                    "Filtering not applied",
                    "Not enough chunks in smallest document",
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
        source_files_to_collections_map: Dict,
        progress: Optional[Progress] = None,
        pbar_generate_contexts_id: Optional[int] = None,
    ):
        contexts_per_doc, scores_per_doc = (
            await self._a_get_n_random_contexts_per_source_file(
                path=path,
                n_contexts_per_source_file=num_context_per_source_file,
                context_size=max_context_size,
                similarity_threshold=self.similarity_threshold,
                source_files_to_collections_map=source_files_to_collections_map,
                progress=progress,
                pbar_generate_contexts_id=pbar_generate_contexts_id,
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
        source_files_to_collections_map: Dict,
        progress: Optional[Progress] = None,
        pbar_generate_contexts_id: Optional[int] = None,
    ):
        assert (
            n_contexts_per_source_file > 0
        ), "n_contexts_per_doc must be a positive integer."
        assert context_size > 0, "context_size must be a positive integer."
        assert (
            0 <= similarity_threshold <= 1
        ), "similarity_threshold must be between 0 and 1."

        contexts = []
        scores = []
        num_query_docs = 0
        collection = source_files_to_collections_map[path]
        random_chunks, scores = self._get_n_random_chunks_per_source_file(
            path=path,
            n_chunks=n_contexts_per_source_file,
            source_files_to_collections_map=source_files_to_collections_map,
            progress=progress,
            pbar_generate_contexts_id=pbar_generate_contexts_id,
        )

        if context_size <= 1:
            return random_chunks, scores

        # Find similar chunks for each context
        for random_chunk in random_chunks:
            pbar_filling_contexts_id = add_pbar(
                progress,
                f"\t\t🔋 Filling context #{self.context_number}",
                (context_size - 1),
            )
            self.pbar_filling_contexts_ids.append(pbar_filling_contexts_id)
            self.context_number += 1
            context = [random_chunk]
            if not random_chunk.strip():
                update_pbar(
                    progress,
                    pbar_filling_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                continue

            similar_chunks = collection.query(
                self.embedder.embed_text(random_chunk), n_results=context_size
            )
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            if len(similar_chunk_texts) <= 1:
                update_pbar(
                    progress,
                    pbar_filling_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                continue
            else:
                similar_chunk_texts = similar_chunk_texts[1:]
            for j, similar_chunk_text in enumerate(similar_chunk_texts):
                similar_chunk_similarity_score = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity_score > similarity_threshold
                ):
                    context.append(similar_chunk_text)
                update_pbar(progress, pbar_filling_contexts_id, remove=False)
                update_pbar(progress, pbar_generate_contexts_id, remove=False)
            update_pbar(
                progress,
                pbar_generate_contexts_id,
                remove=False,
                advance=context_size - 1 - len(similar_chunk_texts),
            )
            contexts.append(context)

        return contexts, scores

    async def _a_get_n_random_contexts_per_source_file(
        self,
        path: str,
        n_contexts_per_source_file: int,
        context_size: int,
        similarity_threshold: float,
        source_files_to_collections_map: Dict,
        progress: Optional[Progress] = None,
        pbar_generate_contexts_id: Optional[int] = None,
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
        random_chunks, scores = (
            await self._a_get_n_random_chunks_per_source_file(
                path=path,
                n_chunks=n_contexts_per_source_file,
                source_files_to_collections_map=source_files_to_collections_map,
                progress=progress,
                pbar_generate_contexts_id=pbar_generate_contexts_id,
            )
        )

        if context_size <= 1:
            return random_chunks, scores

        # Find similar chunks for each context
        for random_chunk in random_chunks:
            pbar_filling_contexts_id = add_pbar(
                progress,
                f"\t\t🔋 Filling context #{self.context_number}",
                (context_size - 1),
            )
            self.pbar_filling_contexts_ids.append(pbar_filling_contexts_id)
            self.context_number += 1
            context = [random_chunk]
            if not random_chunk.strip():
                update_pbar(
                    progress,
                    pbar_filling_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                continue

            similar_chunks = collection.query(
                await self.embedder.a_embed_text(random_chunk),
                n_results=context_size,
            )
            similar_chunk_texts = similar_chunks["documents"][num_query_docs]
            if len(similar_chunk_texts) <= 1:
                update_pbar(
                    progress,
                    pbar_filling_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=context_size - 1,
                    remove=False,
                )
                continue
            else:
                similar_chunk_texts = similar_chunk_texts[1:]

            for j, similar_chunk_text in enumerate(similar_chunk_texts):
                similar_chunk_similarity_score = (
                    1 - similar_chunks["distances"][num_query_docs][j]
                )
                if (
                    similar_chunk_text not in context
                    and similar_chunk_similarity_score > similarity_threshold
                ):
                    context.append(similar_chunk_text)
                update_pbar(progress, pbar_filling_contexts_id, remove=False)
                update_pbar(progress, pbar_generate_contexts_id, remove=False)
            update_pbar(
                progress,
                pbar_generate_contexts_id,
                remove=False,
                advance=context_size - 1 - len(similar_chunk_texts),
            )
            contexts.append(context)

        return contexts, scores

    #########################################################
    ### Get Random Chunks ###################################
    #########################################################

    def _get_n_random_chunks_per_source_file(
        self,
        path: str,
        n_chunks: int,
        source_files_to_collections_map: Dict,
        progress: Optional[Progress] = None,
        pbar_generate_contexts_id: Optional[int] = None,
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
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=self.max_retries,
                    remove=False,
                )
            return chunks, scores

        # Evaluate sampled chunks
        evaluated_chunks = []
        scores = []
        retry_count = 0
        for chunk in chunks:
            score = self.evaluate_chunk(chunk)
            if score > self.filter_threshold:
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=self.max_retries - retry_count,
                    remove=False,
                )
                evaluated_chunks.append(chunk)
                scores.append(score)
                retry_count = 0
            else:
                update_pbar(progress, pbar_generate_contexts_id, remove=False)
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
        source_files_to_collections_map: Dict,
        progress: Optional[Progress] = None,
        pbar_generate_contexts_id: Optional[int] = None,
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
                update_pbar(
                    progress,
                    pbar_generate_contexts_id,
                    advance=self.max_retries,
                    remove=False,
                )
                return await self.a_evaluate_chunk(chunk)

            scores = await asyncio.gather(
                *(update_and_evaluate(chunk) for chunk in chunks)
            )
            return chunks, scores

        # Evaluate sampled chunks
        async def a_evaluate_chunk_and_update(chunk):
            score = await self.a_evaluate_chunk(chunk)
            update_pbar(progress, pbar_generate_contexts_id, remove=False)
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
                f"Adjust the `min_context_length` to no more than {collection_size}, or reduce `chunk_size`.",
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

    def _load_docs(
        self,
        progress: Optional[Progress] = None,
        pbar_load_docs_id: Optional[int] = None,
    ):
        doc_to_chunker_map = {}
        for path in self.document_paths:
            doc_chunker = DocumentChunker(self.embedder)
            doc_chunker.load_doc(path, self.encoding)
            doc_to_chunker_map[path] = doc_chunker
            update_pbar(progress, pbar_load_docs_id, remove=False)
        return doc_to_chunker_map

    async def _a_load_docs(
        self,
        progress: Optional[Progress] = None,
        pbar_load_docs_id: Optional[int] = None,
    ):
        doc_to_chunker_map = {}

        async def a_process_document(
            path: str,
            progress: Optional[Progress] = None,
            pbar_load_docs_id: Optional[int] = None,
        ):
            doc_chunker = DocumentChunker(self.embedder)
            await doc_chunker.a_load_doc(path, self.encoding)
            doc_to_chunker_map[path] = doc_chunker
            update_pbar(progress, pbar_load_docs_id, remove=False)

        tasks = [
            a_process_document(path, progress, pbar_load_docs_id)
            for path in self.document_paths
        ]
        await asyncio.gather(*tasks)
        return doc_to_chunker_map
