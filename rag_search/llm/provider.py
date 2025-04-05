from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            messages: List of message objects with role and content
            system_prompt: Optional system prompt to override default
            temperature: Optional temperature parameter for generation
            max_tokens: Optional maximum tokens to generate
            
        Returns:
            Generated response text
        """
        pass
    
    @property
    @abstractmethod
    def context_window(self) -> int:
        """Get the context window size for the LLM."""
        pass

class LLMException(Exception):
    """Exception raised for LLM-related errors."""
    pass
