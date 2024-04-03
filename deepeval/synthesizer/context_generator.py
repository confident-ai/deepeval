from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import chromadb
import random

from deepeval.synthesizer.doc_chunker import (
    DocumentChunker,
    Chunk,
    get_embedding_similarity,
)
from deepeval.models.openai_embedding_model import OpenAIEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel

DB_FOLDER = ".vectordb"
DB_COLLECTION_NAME = "synth_vectordb"

class ContextGenerator:
    def __init__(
        self,
        document_paths: List[str],
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        multithreading: bool = False,
    ):

        self.embedder: DeepEvalBaseEmbeddingModel = OpenAIEmbeddingModel()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.multithreading = multithreading
        self.document_paths: List[str] = document_paths

        self.client = chromadb.PersistentClient(path=DB_FOLDER)
        # TODO: Potential bug, calling generate_goldens_from_docs
        # twice in a notebook enviornment will not refresh combined chunks
        self.combined_chunks: List[Chunk] = self._load_docs()

    ############### Generate Topics ########################
    def generate_contexts(
        self, num_context: int, max_context_size: int = 3
    ) -> Tuple[List[List[str]], List[str]]:
        num_chunks = len(self.combined_chunks)
        num_context = min(num_context, num_chunks)
        clusters = self._get_n_random_clusters(
            n=num_context, cluster_size=max_context_size
        )
        contexts: List[List[str]] = []
        source_files = []
        for cluster in clusters:
            context = [chunk.content for chunk in cluster]
            contexts.append(context)
            source_files.append(cluster[0].source_file)
        return contexts, source_files

    ############### Load Docs #############################
    def _load_docs(self) -> List[Chunk]:
        # check if docs are already in db
        new_doc_paths = []
        old_doc_paths = []
        collection = self.client.get_or_create_collection(name="test")
        for path in self.document_paths:
            docs_from_path = collection.get(where={"source": path})
            if len(docs_from_path['documents']) == 0:
                new_doc_paths.append(path)
            else:
                old_doc_paths.append(path)
        # create chunks for docs not in db
        if len(new_doc_paths) != 0:
            print("Calculating embeddings for: " + str(new_doc_paths))
        combined_chunks = []
        if not self.multithreading:
             for path in new_doc_paths:
                doc_chunker = DocumentChunker(
                    self.embedder, self.chunk_size, self.chunk_overlap
                )
                chunks = doc_chunker.load_doc(path)
                combined_chunks.extend(chunks)
        else:
            with ThreadPoolExecutor() as executor:
                future_to_path = {
                    executor.submit(self.process_document, path): path
                    for path in new_doc_paths
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        chunks = future.result()
                        combined_chunks.extend(chunks)
                    except Exception as exc:
                        print(f"{path} generated an exception: {exc}")
        self.save_chunks_to_db(collection, combined_chunks)
        # create chunks from docs in db
        saved_chunks = self.get_chunks_from_db(collection, old_doc_paths)
        combined_chunks.extend(saved_chunks)
        return combined_chunks


    def process_document(self, path):
            doc_chunker = DocumentChunker(
                self.embedder, self.chunk_size, self.chunk_overlap
            )
            return doc_chunker.load_doc(path)
    

    def save_chunks_to_db(self, collection, combined_chunks):
        if len(combined_chunks) == 0: return
        documents = []
        embeddings = []
        metadatas=[]
        ids=[]
        for i, chunk in enumerate(combined_chunks):
            documents.append(chunk.content)
            embeddings.append(chunk.embedding)
            metadatas.append({"source": chunk.source_file, 
                             "similarity_to_mean": chunk.similarity_to_mean})
            ids.append(chunk.source_file + str(i))
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def get_chunks_from_db(self, collection, paths):
        chunks = []
        filter = {}
        if len(paths) == 0:
            return []
        elif len(paths) == 1:
            filter["source"] = paths[0]
        else:
            filter["$or"] = [{"source": {"$eq": path}} for path in paths]
        print("Getting cached embeddings for: " + str(paths))
        results = collection.get(where=filter, include=['embeddings', 'documents', 'metadatas'])
        for i in range(len(results["ids"])):
            chunk = Chunk(
                id=results['ids'][i],
                content=results['documents'][i],
                embedding=results['embeddings'][i],
                source_file=results['metadatas'][i]['source'],
                similarity_to_mean=results['metadatas'][i]['similarity_to_mean']
            )
            chunks.append(chunk)      
        return chunks


    ############### Search N Chunks ########################
    def _get_n_random_clusters(
        self, n: int, cluster_size: int
    ) -> List[List[Chunk]]:
        chunk_cluster = []
        random_chunks = self._get_n_random_chunks(n)
        for chunk in random_chunks:
            cluster = [chunk]
            cluster.extend(
                self._get_n_other_similar_chunks(chunk, n=cluster_size - 1)
            )
            chunk_cluster.append(cluster)
        return chunk_cluster

    def _get_n_random_chunks(self, n: int) -> List[str]:
        n = min(len(self.combined_chunks), n)
        return random.sample(self.combined_chunks, n)

    def _get_n_other_similar_chunks(
        self,
        query_chunk: Chunk,
        n: int,
        threshold: float = 0.7,
    ) -> List[Chunk]:
        if not self.combined_chunks or len(self.combined_chunks) < n:
            raise ValueError(
                "Not enough chunks to return the requested number of random nodes."
            )
        query_embedding = self.embedder.embed_query(query_chunk.content)
        similar_chunks = None

        similarities = []
        filtered_indices = []

        for i, c in enumerate(self.combined_chunks):
            sim = get_embedding_similarity(query_embedding, (c.embedding))
            similarities.append(sim)

            if sim > threshold and self.combined_chunks[i].id != query_chunk.id:
                filtered_indices.append(i)

        sorted_indices = sorted(
            filtered_indices, key=lambda i: similarities[i], reverse=True
        )
        top_n_indices = sorted_indices[:n]
        similar_chunks = [self.combined_chunks[i] for i in top_n_indices]

        return similar_chunks

########################################

import time

if __name__ == "__main__":
    start_time = time.time()
    generator = ContextGenerator(document_paths=['data/pdf_example.pdf', 'data/txt_example.txt', 'data/docx_example.docx', 'data/large.pdf'])
    end_time = time.time()
    
    print(f"Initialization and loading took {end_time - start_time:.2f} seconds.")
