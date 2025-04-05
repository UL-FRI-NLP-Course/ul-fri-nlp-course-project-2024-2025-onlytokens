from typing import List, Dict, Any, Optional
import os
import asyncio
import json
from termcolor import colored

from rag_search.llm.provider import LLMProvider, LLMException
from rag_search.llm.prompts import DEFAULT_SYSTEM_PROMPT
from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info,
    log_input, log_output, log_success, log_error
)

class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API to connect to OpenAI models."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        context_length: int = 4096,
        verbose: bool = False,
        default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
            api_base: Base URL for API (optional)
            context_length: Maximum context length
            verbose: Whether to print verbose output
            default_system_prompt: Default system prompt to use
            **kwargs: Additional parameters to pass to OpenAI API
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
            
        self._context_length = context_length
        self.default_system_prompt = default_system_prompt
        self.verbose = verbose
        self.additional_kwargs = kwargs
        
        # Set up OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # Test connection if possible
        if self.verbose:
            log_info(f"LLM ready! Using model: {self.model}", "LLM")
    
    @property
    def context_window(self) -> int:
        """Get the context window size for the LLM."""
        return self._context_length
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1024
    ) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            messages: List of message objects with role and content
            system_prompt: Optional system prompt to override default
            temperature: Optional temperature parameter
            max_tokens: Optional maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # If system prompt is provided, add it at the beginning
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
       
        # Log request if verbose
        if self.verbose:
            log_operation_start("GENERATING RESPONSE", "LLM")
            log_info(f"Using model: {self.model}", "LLM")
            
            # Log the last user message as input
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    log_input(f"Query: {content}", "LLM")
                    break
        
        # Run the model in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **self.additional_kwargs
                )
            )
            
            # Extract the response text
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                
                # Log response if verbose
                if self.verbose:
                    # Log token usage
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = response.usage.prompt_tokens
                        completion_tokens = response.usage.completion_tokens
                        total_tokens = response.usage.total_tokens
                        log_info(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total", "LLM")
                    
                    # Log full response without truncation
                    log_output(f"Response: {result}", "LLM")
                    log_success(f"Generated {len(result)} chars", "LLM")
                    log_operation_end("GENERATING RESPONSE", "LLM")
                
                return result
            else:
                raise LLMException("Failed to get a valid response from the model")
                
        except Exception as e:
            if self.verbose:
                log_error(f"Generation failed: {str(e)}", "LLM")
                log_operation_end("GENERATING RESPONSE", "LLM")
            raise LLMException(f"Error generating response with OpenAI: {str(e)}") 