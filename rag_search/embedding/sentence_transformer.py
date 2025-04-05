from typing import Dict, List, Any, Union, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info,
    log_input, log_output, log_success, log_error, log_warning
)

class SentenceTransformerEmbedder:
    """Text embedder using SentenceTransformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the sentence transformer embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to use for inference (cpu, cuda, etc.)
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings to unit length
            verbose: Whether to print verbose logs
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.verbose = verbose
        
        # Use CPU if no device is specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load the model
        try:
            if self.verbose:
                log_operation_start("LOADING EMBEDDING MODEL", "SentenceTransformerEmbedder")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            
            if self.verbose:
                log_success(f"Loaded model '{model_name}' on {self.device}", "SentenceTransformerEmbedder")
                log_operation_end("LOADING EMBEDDING MODEL", "SentenceTransformerEmbedder")
                log_info("Embedder ready!", "SentenceTransformerEmbedder")
        except Exception as e:
            if self.verbose:
                log_error("Failed to load embedding model", "SentenceTransformerEmbedder", e)
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            query: The query text to embed
            
        Returns:
            Query embedding as a numpy array
        """
        if self.verbose:
            log_operation_start("EMBEDDING QUERY", "SentenceTransformerEmbedder") 
            log_input(f"Query text ({len(query)} chars)", "SentenceTransformerEmbedder")
        
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=self.normalize_embeddings
            )
            
            if self.verbose:
                log_success(f"Generated query embedding of shape {embedding.shape}", "SentenceTransformerEmbedder")
                log_operation_end("EMBEDDING QUERY", "SentenceTransformerEmbedder")
            
            return embedding
        except Exception as e:
            if self.verbose:
                log_error("Failed to embed query", "SentenceTransformerEmbedder", e)
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings as a numpy array
        """
        if not texts:
            if self.verbose:
                log_warning("No texts provided for embedding", "SentenceTransformerEmbedder")
            return np.array([])
        
        if self.verbose:
            log_operation_start("EMBEDDING TEXTS", "SentenceTransformerEmbedder")
            log_input(f"{len(texts)} texts to embed", "SentenceTransformerEmbedder")
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings
            )
            
            if self.verbose:
                log_success(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}", "SentenceTransformerEmbedder")
                log_operation_end("EMBEDDING TEXTS", "SentenceTransformerEmbedder")
            
            return embeddings
        except Exception as e:
            if self.verbose:
                log_error("Failed to embed texts", "SentenceTransformerEmbedder", e)
            raise
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents with text content.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of document dictionaries with added 'embedding' field
        """
        if not documents:
            if self.verbose:
                log_warning("No documents provided for embedding", "SentenceTransformerEmbedder")
            return []
        
        if self.verbose:
            log_operation_start("EMBEDDING DOCUMENTS", "SentenceTransformerEmbedder")
            log_input(f"{len(documents)} documents to embed", "SentenceTransformerEmbedder")
            
        # Extract texts for embedding
        texts = [doc.get('content', '') for doc in documents]
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Add embeddings to documents
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc['embedding'] = embedding
                
            if self.verbose:
                log_success(f"Added embeddings to {len(documents)} documents", "SentenceTransformerEmbedder")
                log_output(f"Document embeddings of shape {embeddings.shape}", "SentenceTransformerEmbedder")
                log_operation_end("EMBEDDING DOCUMENTS", "SentenceTransformerEmbedder")
                
            return documents
        except Exception as e:
            if self.verbose:
                log_error("Failed to embed documents", "SentenceTransformerEmbedder", e)
            raise 