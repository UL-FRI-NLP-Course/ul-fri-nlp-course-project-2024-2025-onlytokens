from typing import List, Dict, Any, Optional
import os
import asyncio
from pathlib import Path

from rag_search.llm.provider import LLMProvider, LLMException
from rag_search.llm.prompts import DEFAULT_SYSTEM_PROMPT

class LocalLLMProvider(LLMProvider):
    """LLM provider using local models with llama.cpp."""
    
    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        """
        Initialize local LLM provider.
        
        Args:
            model_path: Path to the GGUF model file
            context_length: Maximum context length for the model
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of threads to use for CPU inference
            verbose: Whether to print verbose output
            default_system_prompt: Default system prompt to use
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")
            
        model_path = os.path.expanduser(model_path)
        if not os.path.exists(model_path):
            raise LLMException(f"Model file not found: {model_path}")
            
        # Configure number of threads if not specified
        if n_threads is None:
            import multiprocessing
            n_threads = max(1, multiprocessing.cpu_count() // 2)
            
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=verbose
            )
            self._context_length = context_length
            self.default_system_prompt = default_system_prompt
            self.model_path = model_path
        except Exception as e:
            raise LLMException(f"Failed to initialize local LLM: {str(e)}")
    
    @property
    def context_window(self) -> int:
        """Get the context window size for the LLM."""
        return self._context_length
    
    async def generate(
        self,
        context: str,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1024
    ) -> str:
        """
        Generate a response using the local LLM.
        
        Args:
            context: Context to provide to the LLM
            query: Query to answer
            system_prompt: Optional system prompt to override default
            temperature: Optional temperature parameter
            max_tokens: Optional maximum tokens to generate
            
        Returns:
            Generated response text
        """
        prompt = system_prompt or self.default_system_prompt
        
        # Create a chat-format prompt
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Run the model in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            )
            
            # Extract the response text
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"].strip()
            else:
                raise LLMException("Failed to get a valid response from the model")
                
        except Exception as e:
            raise LLMException(f"Error generating response: {str(e)}")
