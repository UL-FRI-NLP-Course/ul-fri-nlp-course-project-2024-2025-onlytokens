from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic, Union

T = TypeVar('T')

class SearchException(Exception):
    """Base exception for search-related errors."""
    pass

class SearchResult(Generic[T]):
    """Container for search results with error handling."""
    
    def __init__(self, data: Optional[T] = None, error: Optional[str] = None):
        self.data = data
        self.error = error
        self.success = error is None

    @property
    def failed(self) -> bool:
        return not self.success

class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 8,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a search and return results.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            location: Optional location to constrain search
            
        Returns:
            Dictionary containing search results
        """
        pass
