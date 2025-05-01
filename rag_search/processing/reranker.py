from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_data, log_error, log_success, log_chunks, log_warning
)

class Reranker(ABC):
    """Base class for content reranking."""
    
    @abstractmethod
    def rerank(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank content chunks based on relevance to query.
        
        Args:
            chunks: List of content chunks
            query: Query to rank against
            
        Returns:
            Reranked list of content chunks
        """
        pass

class JinaAIReranker(Reranker):
    """Reranker using Jina AI's multilingual reranker model."""
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        top_k: int = 5,
        score_threshold: float = 0.0,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        verbose: bool = False
    ):
        """
        Initialize Jina AI reranker.
        
        Args:
            model_name: Name of the Jina AI model to use
            top_k: Number of top results to return
            score_threshold: Minimum score to include a result
            batch_size: Batch size for scoring
            max_length: Maximum sequence length
            device: Device to run model on ('cuda', 'mps', or 'cpu')
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        
        if self.verbose:
            log_operation_start("INITIALIZE JINA AI RERANKER", "JinaAI")
            log_data("Model", model_name, "JinaAI")
            log_data("Top-K", top_k, "JinaAI")
            log_data("Score threshold", score_threshold, "JinaAI")
            log_data("Batch size", batch_size, "JinaAI")
            log_data("Max length", max_length, "JinaAI")
            log_data("Device", device, "JinaAI")
            
        try:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True
            )
            self.model.to(device)
            self.model.eval()
            
            self.top_k = top_k
            self.score_threshold = score_threshold
            self.batch_size = batch_size
            self.max_length = max_length
            self.device = device
            
            if self.verbose:
                log_success("Jina AI model loaded successfully", "JinaAI")
                log_operation_end("INITIALIZE JINA AI RERANKER", "JinaAI")
                
        except ImportError as e:
            if self.verbose:
                log_error("Failed to import required libraries", "JinaAI", e)
                log_operation_end("INITIALIZE JINA AI RERANKER", "JinaAI")
            raise ImportError("Please install transformers and einops: pip install transformers einops")
    
    def rerank(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using Jina AI model.
        
        Args:
            chunks: List of content chunks (no embeddings needed)
            query: Query to rank against
            
        Returns:
            Reranked list of content chunks
        """
        if self.verbose:
            log_operation_start("JINA AI RERANKING", "JinaAI")
            log_data("Chunks to rerank", len(chunks), "JinaAI")
            log_data("Query", query, "JinaAI")
            
        if not chunks:
            if self.verbose:
                log_warning("No chunks to rerank", "JinaAI")
                log_operation_end("JINA AI RERANKING", "JinaAI")
            return []
            
        # Prepare query-chunk pairs for scoring
        pairs = []
        valid_chunks = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if content:
                pairs.append([query, content])
                valid_chunks.append(chunk)
        
        if self.verbose:
            log_data("Valid pairs created", len(pairs), "JinaAI")
                
        if not pairs:
            if self.verbose:
                log_warning("No valid content to rerank", "JinaAI")
                log_operation_end("JINA AI RERANKING", "JinaAI")
            return []
            
        # Score all pairs with Jina AI model
        if self.verbose:
            log_info(f"Scoring pairs with Jina AI model in batches of {self.batch_size}", "JinaAI")
            
        with torch.no_grad():
            scores = self.model.compute_score(
                pairs,
                max_length=self.max_length,
                batch_size=self.batch_size
            )
        
        if self.verbose:
            log_success(f"Scored {len(scores)} pairs", "JinaAI")
            
        # Add scores to chunks
        chunks_with_scores = []
        for chunk, score in zip(valid_chunks, scores):
            score = float(score)
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
            log_data("Chunks with scores above threshold", len(chunks_with_scores), "JinaAI")
            log_success(f"Returning top {len(result)} chunks", "JinaAI")
            log_chunks(result, "JinaAI")
            log_operation_end("JINA AI RERANKING", "JinaAI")
            
        return result


if __name__ == "__main__":
    reranker = JinaAIReranker(verbose=True)
    chunks = [
        {"content": "This is a test chunk."},
        {"content": "Another test chunk."},
        {"content": "Yet another test chunk."},
        {"content": "This is a test chunk."},
        {"content": "Another test chunk."},
        {"content": "Yet another test chunk."},
        {"content": "This is a test chunk."},
        {"content": "Another test chunk."},
        {"content": "Yet another test chunk."},
        {"content": "The capital of France is Paris."},
        ]
    query = "What is the capital of France?"
    result = reranker.rerank(chunks, query)
    print(result)



