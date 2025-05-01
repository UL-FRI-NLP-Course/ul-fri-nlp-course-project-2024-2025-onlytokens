from typing import List, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

from rag_search.processing.embedder import Embedder
from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_data, log_error, log_success, log_chunks, log_warning
)

class Retriever(ABC):
    """Base class for content retrieval."""
    
    @abstractmethod
    def retrieve(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content chunks based on relevance to query.
        
        Args:
            embedded_chunks: List of content chunks with embeddings
            query: Query to retrieve against
            
        Returns:
            Retrieved list of content chunks
        """
        pass

class CosineRetriever(Retriever):
    """Initial retriever using cosine similarity for efficient candidate selection."""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize cosine similarity retriever.
        
        Args:
            embedder: Embedder to use for query embedding (optional)
            top_k: Number of top results to return
            score_threshold: Minimum similarity score to include a result
            verbose: Whether to enable verbose logging
        """
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.verbose = verbose
        
        if self.verbose:
            log_info("CosineRetriever initialized", "Retriever")
            log_data("Top-K", top_k, "Retriever")
            log_data("Score threshold", score_threshold, "Retriever")
            if embedder:
                log_success("Embedder configured for query embedding", "Retriever")
            else:
                log_warning("No embedder provided, using existing embeddings only", "Retriever")
        
    def retrieve(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve initial candidates using cosine similarity.
        
        Args:
            embedded_chunks: List of content chunks with embeddings
            query: Query to retrieve against
            
        Returns:
            Retrieved list of content chunks with highest similarity first
        """
        if self.verbose:
            log_operation_start("RETRIEVING", "Retriever")
            log_data("Chunks to retrieve from", len(embedded_chunks), "Retriever")
            log_data("Query", query, "Retriever")
            
        if not embedded_chunks:
            if self.verbose:
                log_warning("No chunks to retrieve from", "Retriever")
                log_operation_end("RETRIEVING", "Retriever")
            return []
            
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)
        if self.verbose:
            log_info(f"Generated query embedding with dimension {len(query_embedding)}", "Retriever")
        
        # Calculate similarities and sort
        chunks_with_scores = []
        chunks_without_embeddings = 0
        
        for chunk in embedded_chunks:
            if 'embedding' not in chunk:
                chunks_without_embeddings += 1
                continue
                
            similarity = self._cosine_similarity(
                query_embedding,
                chunk['embedding']
            )
            
            if similarity >= self.score_threshold:
                # Create a copy of the chunk with similarity score
                chunk_with_score = dict(chunk)
                chunk_with_score['similarity'] = float(similarity)
                chunks_with_scores.append(chunk_with_score)
        
        if self.verbose:
            if chunks_without_embeddings > 0:
                log_warning(f"Skipped {chunks_without_embeddings} chunks without embeddings", "Retriever")
            log_data("Chunks with scores", len(chunks_with_scores), "Retriever")
                
        # Sort by similarity (highest first)
        sorted_chunks = sorted(
            chunks_with_scores,
            key=lambda x: x.get('similarity', 0.0),
            reverse=True
        )
        
        # Return top k results
        result = sorted_chunks[:self.top_k]
        
        if self.verbose:
            log_success(f"Returning top {len(result)} chunks", "Retriever")
            log_chunks(result, "Retriever")
            log_operation_end("RETRIEVING", "Retriever")
            
        return result
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query string."""
        if self.embedder:
            if self.verbose:
                log_info("Using embedder to generate query embedding", "Retriever")
            return self.embedder.embed_text(query)
            
        # If no embedder provided, we'll need to extract embeddings from the chunks
        if self.verbose:
            log_error("No embedder provided for query embedding", "Retriever")
        raise ValueError("Embedder is required for generating query embedding")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)