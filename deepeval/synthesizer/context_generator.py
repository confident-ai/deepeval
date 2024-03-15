from deepeval.synthesizer.doc_chunker import DocumentChunker, Chunk, Similarity
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import random


class ContextGenerator:
    def __init__(self, 
             document_paths: List[str],
             chunk_size: int = 1024,
             chunk_overlap: int = 0
        ):
        self.document_paths: List[str] = document_paths
        self.combined_chunks: Optional[List[Chunk]] = None
        self.embedder: OpenAIEmbeddings = OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    ############### Generate Topics ########################
    def generate_contexts(self, n_contexts: int = 10, max_context_size: int = 3):
        self._load_docs()
        clusters = self._get_n_random_clusters(n = n_contexts, cluster_size = max_context_size)
        contexts = []
        for cluster in clusters:
            context=[chunk.content for chunk in cluster]
            contexts.append(context)
        return contexts
    
     ############### Load Docs #############################
    def _load_docs(self):
        combined_chunks = []
        for path in self.document_paths:
            doc_chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
            chunks = doc_chunker.load_from_pdf(path)
            combined_chunks.extend(chunks)
        self.combined_chunks = combined_chunks
        return combined_chunks
    
    ############### Search N Chunks ########################
    def _get_n_random_clusters(self, n: int, cluster_size: int):
        chunk_cluster = []
        random_chunks = self._get_n_random_chunks(n)
        for chunk in random_chunks:
            cluster = [chunk]
            cluster.extend(self._get_n_similar_chunks(chunk, n = cluster_size - 1))
            chunk_cluster.append(cluster)
        return chunk_cluster

    def _get_n_random_chunks(self, n: int) -> List[str]:
        if not self.combined_chunks or len(self.combined_chunks) < n:
            raise ValueError("Not enough nodes to return the requested number of random nodes.")
        return random.sample(self.combined_chunks, n)
    
    def _get_n_similar_chunks(
        self, chunk: Chunk, n: int, threshold: float = 0.7, 
    ): 
        embedding = self.embedder.embed_query(chunk.content)
        similarities = [Similarity.get_embedding_similarity(embedding, (c.embedding)) for c in self.combined_chunks]
        filtered_indices = [i for i, sim in enumerate(similarities) if sim > threshold and self.combined_chunks[i].id != chunk.id]
        sorted_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)
        top_n_indices = sorted_indices[:n]
        similar_chunks = [self.combined_chunks[i] for i in top_n_indices]

        return similar_chunks    
    
####################################################
################# Example Usage# ###################
####################################################

# currently only accepts pdf
paths = ["example_data/good_essay_1.pdf", "example_data/good_essay_2.pdf"]
generator = ContextGenerator(paths, chunk_size=1024, chunk_overlap=0)
contexts = generator.generate_contexts()
print("contexts")

