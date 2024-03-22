from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import random
import faiss

from deepeval.synthesizer.doc_chunker import DocumentChunker, Chunk, get_embedding_similarity
from deepeval.models.openai_embedding_model import OpenAIEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel

class ContextGenerator:
    def __init__(self, 
             document_paths: List[str],
             chunk_size: int = 1024,
             chunk_overlap: int = 0,
             fast_mode: bool = True,
             multithreading: bool = False,
        ):
        
        self.embedder: DeepEvalBaseEmbeddingModel = OpenAIEmbeddingModel()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.multithreading = multithreading
        self.document_paths: List[str] = document_paths
        self.combined_chunks: List[Chunk] = self._load_docs()
        
        self.fast_mode = fast_mode
        if fast_mode:
            self.index = self._build_faiss_index()

    ############### Generate Topics ########################
    def generate_contexts(self, num_context: int, max_context_size: int = 2) -> List[List[str]]:
        
        num_chunks = len(self.combined_chunks)
        if num_context > num_chunks:
            raise ValueError("Not enough chunks (" + str(num_chunks) + 
                             ") to return the requested number of contexts ("  + str(num_context) + 
                             "). Please decrease chunk_size or increase document length.")
        clusters = self._get_n_random_clusters(n = num_context, cluster_size = max_context_size)
        contexts = []
        for cluster in clusters:
            context=[chunk.content for chunk in cluster]
            contexts.append(context)
        return contexts
    
    
    ############### Load Docs #############################
    def _load_docs(self) -> List[Chunk]:

        def process_document(path):
            doc_chunker = DocumentChunker(self.embedder, self.chunk_size, self.chunk_overlap)
            return doc_chunker.load_doc(path)
        
        combined_chunks = []

        if not self.multithreading:
            for path in self.document_paths:
                doc_chunker = DocumentChunker(self.embedder, self.chunk_size, self.chunk_overlap)
                chunks = doc_chunker.load_doc(path)
                combined_chunks.extend(chunks)
        else:
            with ThreadPoolExecutor() as executor:
                future_to_path = {executor.submit(process_document, path): path for path in self.document_paths}
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        chunks = future.result()
                        combined_chunks.extend(chunks)
                    except Exception as exc:
                        print(f'{path} generated an exception: {exc}')


        return combined_chunks
    
    def _build_faiss_index(self) -> faiss.IndexFlatL2:
        d = len(self.combined_chunks[0].embedding)
        index = faiss.IndexFlatL2(d)
        embeddings = np.array([chunk.embedding for chunk in self.combined_chunks])
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # Add normalized embeddings to the FAISS index
        index.add(normalized_embeddings)
        return index
    
    ############### Search N Chunks ########################
    def _get_n_random_clusters(self, n: int, cluster_size: int) -> List[List[Chunk]]:
        chunk_cluster = []
        random_chunks = self._get_n_random_chunks(n)
        for chunk in random_chunks:
            cluster = [chunk]
            cluster.extend(self._get_n_other_similar_chunks(chunk, n = cluster_size - 1))
            chunk_cluster.append(cluster)
        return chunk_cluster

    def _get_n_random_chunks(self, n: int) -> List[str]:
        if not self.combined_chunks or len(self.combined_chunks) < n:
            raise ValueError("Not enough chunks to return the requested number of random nodes.")
        return random.sample(self.combined_chunks, n)
    
    def _get_n_other_similar_chunks(
        self, query_chunk: Chunk, n: int, threshold: float = 0.7, fast_threshold: float = 0.6
    ) -> List[Chunk]: 
        if not self.combined_chunks or len(self.combined_chunks) < n:
            raise ValueError("Not enough chunks to return the requested number of random nodes.")
        query_embedding = self.embedder.embed_query(query_chunk.content)
        similar_chunks = None

        if not self.fast_mode:
            similarities = []
            filtered_indices = []

            for (i, c) in enumerate(self.combined_chunks):
                sim = get_embedding_similarity(query_embedding, (c.embedding))
                similarities.append(sim)

                if sim > threshold and self.combined_chunks[i].id != query_chunk.id:
                    filtered_indices.append(i)

            sorted_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)
            top_n_indices = sorted_indices[:n]
            similar_chunks = [self.combined_chunks[i] for i in top_n_indices]

        else:
            query_embedding = np.array(query_embedding).reshape(1, -1) 
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            normalized_query_embedding = query_embedding / query_norm
            distances, indices = self.index.search(normalized_query_embedding, n + 1)
            similar_chunks = []
            found = 0  # Counter for found chunks that meet the criteria

            for i in range(len(indices[0])):
                curr_idx = indices[0][i]
                curr_dis = distances[0][i]
                if self.combined_chunks[curr_idx].id == query_chunk.id: 
                    continue  
                if curr_dis > fast_threshold: 
                    continue
                similar_chunks.append(self.combined_chunks[curr_idx])
                found += 1
                if found == n:  # Stop once we have found 'n' similar chunks
                    break

        return similar_chunks    
    
####################################################
################# Example Usage# ###################
####################################################

'''
import time

if __name__ == "__main__":
    paths = ["example_data/txt_example.txt", "example_data/docx_example.docx", "example_data/pdf_example.pdf"]
    
    # Without Fast Mode (FAISS)
    start_no_faiss = time.time()
    generator_no_faiss = ContextGenerator(paths, chunk_size=100, fast_mode=False)
    contexts_no_faiss = generator_no_faiss.generate_contexts(5)
    end_no_faiss = time.time()
    
    contexts_no_faiss_shapes = [len(context) for context in contexts_no_faiss]
    print(f"Shapes of contexts without FAISS: {contexts_no_faiss_shapes}")
    print(f"Time taken without FAISS: {end_no_faiss - start_no_faiss} seconds\n")
    
    # With Fast Mode (FAISS)
    start_with_faiss = time.time()
    generator_with_faiss = ContextGenerator(paths, chunk_size=100, fast_mode=True)
    contexts_with_faiss = generator_with_faiss.generate_contexts(5)
    end_with_faiss = time.time()
    
    contexts_with_faiss_shapes = [len(context) for context in contexts_with_faiss]
    print(f"Shapes of contexts with FAISS: {contexts_with_faiss_shapes}")
    print(f"Time taken with FAISS: {end_with_faiss - start_with_faiss} seconds")

    # With Fast Mode + Multithreading (FAISS)
    start_with_faiss = time.time()
    generator_with_faiss = ContextGenerator(paths, chunk_size=100, fast_mode=True, multithreading=True)
    contexts_with_faiss = generator_with_faiss.generate_contexts(5)
    end_with_faiss = time.time()
    
    contexts_with_faiss_shapes = [len(context) for context in contexts_with_faiss]
    print(f"Shapes of contexts with FAISS + multithreading: {contexts_with_faiss_shapes}")
    print(f"Time taken with FAISS + multithreading: {end_with_faiss - start_with_faiss} seconds")
'''