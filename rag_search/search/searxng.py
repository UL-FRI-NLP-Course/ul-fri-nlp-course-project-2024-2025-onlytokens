import json
import os
from typing import Dict, Any, Optional
import aiohttp

from rag_search.search.base import SearchProvider, SearchResult, SearchException
from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_error, log_success,
    log_search_results
)

class SearXNGException(SearchException):
    """SearXNG-specific exceptions."""
    pass

class SearXNGProvider(SearchProvider):
    """Search provider implementation for SearXNG."""
    
    def __init__(
        self,
        instance_url: Optional[str] = "http://localhost:5555/search",
        api_key: Optional[str] = None,
        default_location: str = "all",
        timeout: int = 10,
        verbose: bool = False
    ):
        """
        Initialize SearXNG provider.
        
        Args:
            instance_url: URL of SearXNG instance (can also use SEARXNG_INSTANCE_URL env var)
            api_key: Optional API key for SearXNG (can also use SEARXNG_API_KEY env var)
            default_location: Default location for searches
            timeout: Request timeout in seconds
            verbose: Whether to enable verbose logging
        """
        self.instance_url = instance_url or os.getenv("SEARXNG_INSTANCE_URL")
        if not self.instance_url:
            raise SearXNGException("SearXNG instance URL not provided and SEARXNG_INSTANCE_URL env var not set")
            
        # Ensure URL ends with /search
        if not self.instance_url.endswith('/search'):
            self.instance_url = self.instance_url.rstrip('/') + '/search'
            
        self.api_key = api_key or os.getenv("SEARXNG_API_KEY")
        self.default_location = default_location
        self.timeout = timeout
        self.verbose = verbose
        
        # Set up request headers
        self.headers = {'Content-Type': 'application/json'}
        if self.api_key:
            self.headers['X-API-Key'] = self.api_key
        
        if self.verbose:
            log_info("Search engine ready!", "SearXNGProvider")
        
    async def search(
        self,
        query: str,
        num_results: int = 8,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a search through SearXNG.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            location: Optional location to constrain search
            
        Returns:
            Dictionary containing search results
        
        Raises:
            SearXNGException: If the search fails
        """
        if not query.strip():
            raise SearXNGException("Search query cannot be empty")
        
        if self.verbose:
            log_operation_start("SEARCHING", "SearXNGProvider")
            log_input(query, "SearXNGProvider")
                
        # Set up request parameters
        #TODO: we should use the search provider's location and time to refine the query ALSO maybe limit the timerange see query enhancer
        params = {
            'q': query,
            'format': 'json',
            'pageno': 1,
            'categories': 'general',
            'language': 'all',
            'safesearch': 0,
            'engines': 'google,bing,duckduckgo',  # Default engines
            'max_results': min(max(1, num_results), 20)  # Reasonable limit
        }
        
        # Add location if provided
        if location and location != 'all':
            params['language'] = location
        elif self.default_location != 'all':
            params['language'] = self.default_location
            
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        
        if self.verbose:
            log_info("Sending request to search engine...", "SearXNGProvider")
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.instance_url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if self.verbose:
                        log_success("Search completed", "SearXNGProvider")
                    
                    # Transform results to a standardized format
                    results = self._format_results(data, num_results)
                    
                    if self.verbose:
                        log_search_results(results, "SearXNGProvider")
                        log_operation_end("SEARCHING", "SearXNGProvider")
                        
                    return results
                
        except aiohttp.ClientError as e:
            if self.verbose:
                log_error("Search failed - API error", "SearXNGProvider", e)
                log_operation_end("SEARCHING", "SearXNGProvider")
            raise SearXNGException(f"SearXNG API request failed: {str(e)}")
        except Exception as e:
            if self.verbose:
                log_error("Search failed - Unexpected error", "SearXNGProvider", e)
                log_operation_end("SEARCHING", "SearXNGProvider")
            raise SearXNGException(f"Unexpected error with SearXNG: {str(e)}")
    
    def _format_results(self, data: Dict[str, Any], num_results: int) -> Dict[str, Any]:
        """Format SearXNG results to a standardized structure."""
        if self.verbose:
            log_info("Processing search results", "SearXNGProvider")
            
        # Extract organic results
        organic_results = []
        #SAVE RESULTS TO A FILE
        with open('search_results.json', 'w') as f:
            json.dump(data, f)
        for result in data.get('results', [])[:num_results]:
            organic_results.append({
                'title': result.get('title', ''),
                'link': result.get('url', ''),
                'snippet': result.get('content', ''),
                'date': result.get('publishedDate', '') #NOTE SOME results have this empty!!!   
            })
            
        # Extract image results
        image_results = []
        for result in data.get('results', []):
            if result.get('img_src'):
                image_results.append({
                    'title': result.get('title', ''),
                    'imageUrl': result.get('img_src', '')
                })
        
        # Format results to a standard structure
        return {
            'organic': organic_results,
            'images': image_results,
            'topStories': [],  # SearXNG doesn't have direct equivalent
            'answerBox': None,  # SearXNG doesn't provide answer box some search providers do!
            'peopleAlsoAsk': None,
            'relatedSearches': data.get('suggestions', []) 
            #TODO we can use these to genarate even beter queries and to do more searches!
            # example response:
            # "suggestions": [
            # "Second hand shop online",
            # "Second Hand shop online slovenija",
            # "Second hand shop Maribor",
            # "Second hand Ljubljana",
            # "Second hand hahaha mnenja",
            # "Second hand shop ptuj",
            # "Second hand trgovina",
            # "Odkup rabljenih oblačil maribor"
            #   ],
        }

if __name__ == "__main__":
    import asyncio
    
    async def test():
        provider = SearXNGProvider(
            verbose=True
        )
        
        results = await provider.search("What is the capital of France?")
        print(json.dumps(results, indent=4, ensure_ascii=False))
    
    asyncio.run(test())

