import numpy as np
from typing import Dict, List, Any, Union, Optional
from sentence_transformers import util

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info,
    log_input, log_output, log_success, log_error
)

class CosineReranker:
    """Reranker based on cosine similarity with query"""
    
    def __init__(
        self,
        top_k: int = 10,
        threshold: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize the cosine similarity reranker.
        
        Args:
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            verbose: Whether to print verbose logs
        """
        self.top_k = top_k
        self.threshold = threshold
        self.verbose = verbose
        
        if self.verbose:
            log_info("Cosine reranker ready!", "CosineReranker")
    
    def rerank(
        self,
        query_embedding: np.ndarray,
        documents: List[Dict[str, Any]],
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on cosine similarity with query embedding.
        
        Args:
            query_embedding: Embedding vector of the query
            documents: List of document dictionaries with 'embedding' field
            include_scores: Whether to include scores in the result
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
            
        if self.verbose:
            log_operation_start("RERANKING WITH COSINE", "CosineReranker")
            log_input(f"{len(documents)} documents to rerank", "CosineReranker")
        
        try:
            # Get document embeddings
            doc_embeddings = np.array([doc.get('embedding') for doc in documents if 'embedding' in doc])
            valid_docs = [doc for doc in documents if 'embedding' in doc]
            
            if len(valid_docs) == 0:
                if self.verbose:
                    log_error("No valid embeddings found in documents", "CosineReranker")
                return []
            
            # Calculate cosine similarities
            # Reshape query embedding to match expected dimensions (1, dim)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Calculate cosine similarities
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0].numpy()
            
            # Create a list of (index, similarity) pairs
            similarity_tuples = [(i, float(sim)) for i, sim in enumerate(similarities)]
            
            # Sort by similarity (descending)
            similarity_tuples.sort(key=lambda x: x[1], reverse=True)
            
            # Apply threshold if needed
            if self.threshold is not None:
                similarity_tuples = [(i, sim) for i, sim in similarity_tuples if sim >= self.threshold]
            
            # Apply top_k limit
            similarity_tuples = similarity_tuples[:self.top_k]
            
            # Create result list
            results = []
            for idx, score in similarity_tuples:
                result = valid_docs[idx].copy()
                if include_scores:
                    result['score'] = score
                results.append(result)
            
            if self.verbose:
                log_success(f"Reranked to {len(results)} documents", "CosineReranker")
                if results:
                    top_score = results[0].get('score', 0) if include_scores else 'N/A'
                    log_output(f"Top similarity score: {top_score:.4f}", "CosineReranker")
                log_operation_end("RERANKING WITH COSINE", "CosineReranker")
            
            return results
        except Exception as e:
            if self.verbose:
                log_error("Error during cosine reranking", "CosineReranker", e)
            raise 