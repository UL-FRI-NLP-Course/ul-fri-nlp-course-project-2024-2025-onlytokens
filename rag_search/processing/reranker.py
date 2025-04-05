from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

from rag_search.processing.embedder import Embedder
from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_data, log_error, log_success, log_chunks
)

class Reranker(ABC):
    """Base class for content reranking."""
    
    @abstractmethod
    def rerank(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank content chunks based on relevance to query.
        
        Args:
            embedded_chunks: List of content chunks with embeddings
            query: Query to rank against
            
        Returns:
            Reranked list of content chunks
        """
        pass

class CosineReranker(Reranker):
    """Reranker using cosine similarity."""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize cosine similarity reranker.
        
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
            log_info("CosineReranker initialized", "Reranker")
            log_data("Top-K", top_k, "Reranker")
            log_data("Score threshold", score_threshold, "Reranker")
            if embedder:
                log_success("Embedder configured for query embedding", "Reranker")
            else:
                log_warning("No embedder provided, using existing embeddings only", "Reranker")
        
    def rerank(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using cosine similarity.
        
        Args:
            embedded_chunks: List of content chunks with embeddings
            query: Query to rank against
            
        Returns:
            Reranked list of content chunks with highest similarity first
        """
        if self.verbose:
            log_operation_start("RERANKING", "Reranker")
            log_data("Chunks to rerank", len(embedded_chunks), "Reranker")
            log_data("Query", query, "Reranker")
            
        if not embedded_chunks:
            if self.verbose:
                log_warning("No chunks to rerank", "Reranker")
                log_operation_end("RERANKING", "Reranker")
            return []
            
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)
        if self.verbose:
            log_info(f"Generated query embedding with dimension {len(query_embedding)}", "Reranker")
        
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
                log_warning(f"Skipped {chunks_without_embeddings} chunks without embeddings", "Reranker")
            log_data("Chunks with scores", len(chunks_with_scores), "Reranker")
                
        # Sort by similarity (highest first)
        sorted_chunks = sorted(
            chunks_with_scores,
            key=lambda x: x.get('similarity', 0.0),
            reverse=True
        )
        
        # Return top k results
        result = sorted_chunks[:self.top_k]
        
        if self.verbose:
            log_success(f"Returning top {len(result)} chunks", "Reranker")
            log_chunks(result, "Reranker")
            log_operation_end("RERANKING", "Reranker")
            
        return result
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query string."""
        if self.embedder:
            if self.verbose:
                log_info("Using embedder to generate query embedding", "Reranker")
            return self.embedder.embed_text(query)
            
        # If no embedder provided, we'll need to extract embeddings from the chunks
        if self.verbose:
            log_error("No embedder provided for query embedding", "Reranker")
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

class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        score_threshold: float = 0.0,
        batch_size: int = 16,
        verbose: bool = False
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            top_k: Number of top results to return
            score_threshold: Minimum score to include a result
            batch_size: Batch size for scoring
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        
        if self.verbose:
            log_operation_start("INITIALIZE CROSS-ENCODER", "CrossEncoder")
            log_data("Model", model_name, "CrossEncoder")
            log_data("Top-K", top_k, "CrossEncoder")
            log_data("Score threshold", score_threshold, "CrossEncoder")
            log_data("Batch size", batch_size, "CrossEncoder")
            
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.top_k = top_k
            self.score_threshold = score_threshold
            self.batch_size = batch_size
            
            if self.verbose:
                log_success("Cross-encoder model loaded successfully", "CrossEncoder")
                log_operation_end("INITIALIZE CROSS-ENCODER", "CrossEncoder")
                
        except ImportError as e:
            if self.verbose:
                log_error("Failed to import sentence-transformers", "CrossEncoder", e)
                log_operation_end("INITIALIZE CROSS-ENCODER", "CrossEncoder")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def rerank(
        self, 
        embedded_chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using cross-encoder model.
        
        Args:
            embedded_chunks: List of content chunks
            query: Query to rank against
            
        Returns:
            Reranked list of content chunks
        """
        if self.verbose:
            log_operation_start("CROSS-ENCODER RERANKING", "CrossEncoder")
            log_data("Chunks to rerank", len(embedded_chunks), "CrossEncoder")
            log_data("Query", query, "CrossEncoder")
            
        if not embedded_chunks:
            if self.verbose:
                log_warning("No chunks to rerank", "CrossEncoder")
                log_operation_end("CROSS-ENCODER RERANKING", "CrossEncoder")
            return []
            
        # Prepare query-chunk pairs for scoring
        pairs = []
        for i, chunk in enumerate(embedded_chunks):
            content = chunk.get('content', '')
            if content:
                pairs.append([query, content])
        
        if self.verbose:
            log_data("Valid pairs created", len(pairs), "CrossEncoder")
                
        if not pairs:
            if self.verbose:
                log_warning("No valid content to rerank", "CrossEncoder")
                log_operation_end("CROSS-ENCODER RERANKING", "CrossEncoder")
            return []
            
        # Score all pairs with cross-encoder
        if self.verbose:
            log_info(f"Scoring pairs with cross-encoder in batches of {self.batch_size}", "CrossEncoder")
            
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        if self.verbose:
            log_success(f"Scored {len(scores)} pairs", "CrossEncoder")
            
        # Add scores to chunks
        chunks_with_scores = []
        pair_idx = 0
        
        for chunk in embedded_chunks:
            if chunk.get('content', ''):
                score = float(scores[pair_idx])
                pair_idx += 1
                
                if score >= self.score_threshold:
                    # Create a copy of the chunk with score
                    chunk_with_score = dict(chunk)
                    chunk_with_score['similarity'] = score
                    chunks_with_scores.append(chunk_with_score)
        
        # Sort by score (highest first)
        sorted_chunks = sorted(
            chunks_with_scores,
            key=lambda x: x.get('similarity', 0.0),
            reverse=True
        )
        
        # Return top k results
        result = sorted_chunks[:self.top_k]
        
        if self.verbose:
            log_data("Chunks with scores above threshold", len(chunks_with_scores), "CrossEncoder")
            log_success(f"Returning top {len(result)} chunks", "CrossEncoder")
            log_chunks(result, "CrossEncoder")
            log_operation_end("CROSS-ENCODER RERANKING", "CrossEncoder")
            
        return result
