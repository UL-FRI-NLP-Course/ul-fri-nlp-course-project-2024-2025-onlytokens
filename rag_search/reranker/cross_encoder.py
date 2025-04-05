from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info,
    log_input, log_output, log_success, log_error
)

class CrossEncoderReranker:
    """Reranker based on cross-encoder model"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to use for inference (cpu, cuda, etc.)
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            batch_size: Batch size for inference
            verbose: Whether to print verbose logs
        """
        self.model_name = model_name
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Use CPU if no device is specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            if self.verbose:
                log_operation_start("LOADING CROSS-ENCODER", "CrossEncoderReranker")
                
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            if self.verbose:
                log_success(f"Loaded model '{model_name}' on {self.device}", "CrossEncoderReranker")
                log_operation_end("LOADING CROSS-ENCODER", "CrossEncoderReranker")
                log_info("Cross-encoder reranker ready!", "CrossEncoderReranker")
        except Exception as e:
            if self.verbose:
                log_error("Failed to load cross-encoder model", "CrossEncoderReranker", e)
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        content_field: str = 'content',
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on cross-encoder scores.
        
        Args:
            query: The search query
            documents: List of document dictionaries with content
            content_field: Field name containing the document text
            include_scores: Whether to include scores in the result
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
            
        if self.verbose:
            log_operation_start("RERANKING WITH CROSS-ENCODER", "CrossEncoderReranker")
            log_input(f"Query: '{query}', {len(documents)} documents", "CrossEncoderReranker")
        
        try:
            # Create query-document pairs
            pairs = [(query, doc.get(content_field, "")) for doc in documents]
            
            # Get scores in batches
            all_scores = []
            
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i+self.batch_size]
                
                # Encode batch
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Get scores
                with torch.no_grad():
                    scores = self.model(**inputs).logits.flatten().cpu().numpy()
                
                all_scores.extend(scores)
            
            # Create a list of (index, score) pairs
            score_tuples = [(i, float(score)) for i, score in enumerate(all_scores)]
            
            # Sort by score (descending)
            score_tuples.sort(key=lambda x: x[1], reverse=True)
            
            # Apply threshold if needed
            if self.threshold is not None:
                score_tuples = [(i, score) for i, score in score_tuples if score >= self.threshold]
            
            # Apply top_k limit
            score_tuples = score_tuples[:self.top_k]
            
            # Create result list
            results = []
            for idx, score in score_tuples:
                result = documents[idx].copy()
                if include_scores:
                    result['score'] = score
                results.append(result)
            
            if self.verbose:
                log_success(f"Reranked to {len(results)} documents", "CrossEncoderReranker")
                if results:
                    top_score = results[0].get('score', 0) if include_scores else 'N/A'
                    log_output(f"Top similarity score: {top_score:.4f}", "CrossEncoderReranker")
                log_operation_end("RERANKING WITH CROSS-ENCODER", "CrossEncoderReranker")
            
            return results
        except Exception as e:
            if self.verbose:
                log_error("Error during cross-encoder reranking", "CrossEncoderReranker", e)
            raise 