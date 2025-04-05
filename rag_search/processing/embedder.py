from typing import List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_data, log_error, log_success, log_embedding_operation,
    log_warning
)

class Embedder(ABC):
    """Base class for text embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
        
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass
        
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of content chunks.
        
        Args:
            chunks: List of content chunks with 'content' field
            
        Returns:
            Same chunks with 'embedding' field added
        """
        # Extract content texts
        texts = [chunk.get('content', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = dict(chunk)
            chunk_with_embedding['embedding'] = embedding
            result.append(chunk_with_embedding)
            
        return result

class SentenceTransformerEmbedder(Embedder):
    """Embedder using Sentence Transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Initialize Sentence Transformer embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to use (cpu, cuda, etc.)
            batch_size: Batch size for embedding multiple texts
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        
        if self.verbose:
            log_operation_start("INITIALIZE EMBEDDER", "SentenceEmbedder")
            log_data("Model", model_name, "SentenceEmbedder")
            log_data("Device", device or "default", "SentenceEmbedder")
            log_data("Batch size", batch_size, "SentenceEmbedder")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.batch_size = batch_size
            
            if self.verbose:
                dim = self.model.get_sentence_embedding_dimension()
                log_success(f"Model loaded successfully with dimension {dim}", "SentenceEmbedder")
                log_operation_end("INITIALIZE EMBEDDER", "SentenceEmbedder")
                
        except ImportError as e:
            if self.verbose:
                log_error("Failed to import sentence-transformers", "SentenceEmbedder", e)
                log_operation_end("INITIALIZE EMBEDDER", "SentenceEmbedder")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.verbose:
            log_operation_start("EMBED TEXT", "SentenceEmbedder")
            preview = text[:50] + "..." if len(text) > 50 else text
            log_data("Text", preview, "SentenceEmbedder")
            
        if not text:
            # Return zero vector for empty text
            dim = self.model.get_sentence_embedding_dimension()
            if self.verbose:
                log_warning("Empty text provided, returning zero vector", "SentenceEmbedder")
                log_operation_end("EMBED TEXT", "SentenceEmbedder")
            return [0.0] * dim
            
        # Generate embedding
        embedding = self.model.encode(text, show_progress_bar=False)
        
        # Convert to list of floats
        result = embedding.tolist()
        
        if self.verbose:
            log_success(f"Embedding generated with dimension {len(result)}", "SentenceEmbedder")
            log_operation_end("EMBED TEXT", "SentenceEmbedder")
            
        return result
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.verbose:
            log_operation_start("EMBED TEXTS", "SentenceEmbedder")
            log_embedding_operation(len(texts), "SentenceEmbedder")
        
        # Handle empty texts with zero vectors
        dim = self.model.get_sentence_embedding_dimension()
        result = []
        
        # Identify valid texts and their positions
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text:
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                result.append([0.0] * dim)
        
        if self.verbose:
            log_data("Valid texts", f"{len(valid_texts)}/{len(texts)}", "SentenceEmbedder")
            if len(texts) - len(valid_texts) > 0:
                log_warning(f"{len(texts) - len(valid_texts)} empty texts will get zero vectors", "SentenceEmbedder")
                
        if not valid_texts:
            if self.verbose:
                log_warning("No valid texts to embed", "SentenceEmbedder")
                log_operation_end("EMBED TEXTS", "SentenceEmbedder")
            return result
            
        # Generate embeddings for valid texts
        if self.verbose:
            log_info(f"Generating embeddings in batches of {self.batch_size}", "SentenceEmbedder")
            
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Place embeddings in the correct positions
        for i, embedding in zip(valid_indices, embeddings):
            # Insert at the appropriate position
            while len(result) <= i:
                result.append(None)
            result[i] = embedding.tolist()
        
        if self.verbose:
            log_success(f"Generated {len(valid_texts)} embeddings with dimension {dim}", "SentenceEmbedder")
            log_operation_end("EMBED TEXTS", "SentenceEmbedder")
            
        return result
        
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of content chunks.
        
        Args:
            chunks: List of content chunks with 'content' field
            
        Returns:
            Same chunks with 'embedding' field added
        """
        if self.verbose:
            log_operation_start("EMBED CHUNKS", "SentenceEmbedder")
            log_data("Chunks", f"{len(chunks)} to embed", "SentenceEmbedder")
            
        # Extract content texts
        texts = [chunk.get('content', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = dict(chunk)
            chunk_with_embedding['embedding'] = embedding
            result.append(chunk_with_embedding)
        
        if self.verbose:
            log_success(f"Added embeddings to {len(result)} chunks", "SentenceEmbedder")
            log_operation_end("EMBED CHUNKS", "SentenceEmbedder")
            
        return result

class HuggingFaceEmbedder(Embedder):
    """Embedder using Hugging Face Transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize Hugging Face embedder.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to use (auto, cpu, cuda, etc.)
            batch_size: Batch size for embedding multiple texts
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Set device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.batch_size = batch_size
            
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
            
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to create sentence embeddings."""
        import torch
        
        # First element of model_output contains token embeddings
        token_embeddings = model_output[0]
        
        # Calculate attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum token embeddings and divide by attention mask sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
            
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        import torch
        
        if not text:
            # Get embedding dimension from model
            with torch.no_grad():
                dummy_output = self.model(
                    **self.tokenizer("dummy text", return_tensors="pt").to(self.device)
                )
                dim = dummy_output.last_hidden_state.shape[-1]
            
            return [0.0] * dim
            
        # Encode text
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            model_output = self.model(**encoded_input)
            embedding = self._mean_pooling(model_output, encoded_input["attention_mask"])
            
        # Convert to list and return
        return embedding[0].cpu().numpy().tolist()
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        import torch
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Skip empty texts
            valid_texts = []
            valid_indices = []
            batch_embeddings = [None] * len(batch_texts)
            
            for j, text in enumerate(batch_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(j)
                else:
                    # Get embedding dimension from model
                    with torch.no_grad():
                        dummy_output = self.model(
                            **self.tokenizer("dummy text", return_tensors="pt").to(self.device)
                        )
                        dim = dummy_output.last_hidden_state.shape[-1]
                    
                    batch_embeddings[j] = [0.0] * dim
            
            if valid_texts:
                # Encode valid texts
                with torch.no_grad():
                    encoded_input = self.tokenizer(
                        valid_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    model_output = self.model(**encoded_input)
                    embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
                    
                    # Place embeddings in the correct positions
                    for idx, j in enumerate(valid_indices):
                        batch_embeddings[j] = embeddings[idx].cpu().numpy().tolist()
            
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
