import asyncio
from typing import Dict, List, Any, Optional, Union
import aiohttp
from bs4 import BeautifulSoup
from dataclasses import dataclass

from rag_search.scraping.strategies import ExtractionStrategy, NoExtractionStrategy
from rag_search.utils.async_utils import gather_with_concurrency
from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_error, log_success, log_warning
)

@dataclass
class ScrapingResult:
    """Results from web scraping."""
    url: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class WebScraper:
    """Web scraper with support for multiple extraction strategies."""
    
    def __init__(
        self,
        strategies: List[str] = ["no_extraction"],
        user_agent: str = "Mozilla/5.0 (compatible; RAGSearch/1.0)",
        timeout: int = 10,
        max_concurrency: int = 5,
        debug: bool = False
    ):
        """
        Initialize the web scraper.
        
        Args:
            strategies: List of extraction strategy names to use
            user_agent: User agent string for HTTP requests
            timeout: Request timeout in seconds
            max_concurrency: Maximum number of concurrent requests
            debug: Whether to enable debug output
        """
        self.strategies = strategies
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.debug = debug
        
        # Initialize strategy objects
        self.strategy_objects = self._init_strategies(strategies)
        
        if debug:
            log_info("Web scraper ready!", "WebScraper")
        
    def _init_strategies(self, strategy_names: List[str]) -> Dict[str, ExtractionStrategy]:
        """Initialize strategy objects based on strategy names."""
        from rag_search.scraping.strategies import (
            NoExtractionStrategy,
            CSSExtractionStrategy,
            XPathExtractionStrategy
        )
        
        strategy_map = {
            "no_extraction": NoExtractionStrategy(),
            "css": CSSExtractionStrategy(),
            "xpath": XPathExtractionStrategy(),
            # Additional strategies can be added here
        }
        
        # Validate strategies
        invalid_strategies = set(strategy_names) - set(strategy_map.keys())
        if invalid_strategies:
            raise ValueError(f"Invalid extraction strategies: {invalid_strategies}")
            
        # Return only the requested strategies
        return {name: strategy_map[name] for name in strategy_names if name in strategy_map}
        
    async def scrape(self, url: str) -> Dict[str, ScrapingResult]:
        """
        Scrape a single URL using all configured strategies.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary mapping strategy names to scraping results
        """
        if self.debug:
            log_operation_start("SCRAPING", "WebScraper")
            log_input(url, "WebScraper")
            
        # Handle Wikipedia URLs specially
        if 'wikipedia.org/wiki/' in url:
            results = await self._handle_wikipedia(url)
            if self.debug:
                self._log_scraping_results(results, url)
                log_operation_end("SCRAPING", "WebScraper")
            return results
        
        # For regular URLs, scrape content
        try:
            # Fetch HTML content
            html_content = await self._fetch_html(url)
            if self.debug:
                log_info(f"Downloaded {len(html_content)} bytes", "WebScraper")
                
            # Apply each strategy to the content
            results = {}
            for name, strategy in self.strategy_objects.items():
                try:
                    if self.debug:
                        log_info(f"Extracting with {name}...", "WebScraper")
                        
                    extracted_content = strategy.extract(html_content, url)
                    if self.debug:
                        content_preview = (extracted_content[:60] + "...") if len(extracted_content) > 60 else extracted_content
                        log_success(f"Got {len(extracted_content)} chars", "WebScraper")
                    
                    results[name] = ScrapingResult(
                        url=url,
                        success=True,
                        content=extracted_content,
                        metadata={"strategy": name}
                    )
                except Exception as e:
                    if self.debug:
                        log_error(f"Failed with {name}", "WebScraper", e)
                    results[name] = ScrapingResult(
                        url=url,
                        success=False,
                        error=f"Extraction error: {str(e)}",
                        metadata={"strategy": name}
                    )
            
            if self.debug:
                self._log_scraping_results(results, url)
                log_operation_end("SCRAPING", "WebScraper")
                
            return results
            
        except Exception as e:
            # If overall scraping fails, return error results for all strategies
            if self.debug:
                log_error(f"Scraping failed", "WebScraper", e)
                log_operation_end("SCRAPING", "WebScraper")
                
            return {
                name: ScrapingResult(
                    url=url,
                    success=False,
                    error=f"Scraping error: {str(e)}",
                    metadata={"strategy": name}
                ) for name in self.strategy_objects.keys()
            }
    
    async def scrape_many(self, urls: List[str]) -> List[Dict[str, ScrapingResult]]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of dictionaries mapping strategy names to scraping results
        """
        if self.debug:
            log_operation_start("BATCH SCRAPING", "WebScraper")
            log_input(f"{len(urls)} URLs", "WebScraper")
        
        tasks = [self.scrape(url) for url in urls]
        results = await gather_with_concurrency(self.max_concurrency, *tasks)
        
        if self.debug:
            success_count = sum(1 for r in results for strategy_result in r.values() if strategy_result.success)
            total_count = sum(len(r) for r in results)
            log_success(f"Scraped {success_count}/{total_count} resources", "WebScraper")
            log_operation_end("BATCH SCRAPING", "WebScraper")
            
        return results
    
    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from a URL."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        if self.debug:
            log_info(f"Downloading {url}", "WebScraper")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                content = await response.text()
                return content
    
    async def _handle_wikipedia(self, url: str) -> Dict[str, ScrapingResult]:
        """Special handling for Wikipedia URLs."""
        if self.debug:
            log_info(f"Wikipedia article detected", "WebScraper")
            
        try:
            # We could implement a more efficient Wikipedia extractor here
            # For now, we'll use the regular scraping method
            html_content = await self._fetch_html(url)
            
            # Create a specialized Wikipedia extractor
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract the main content area
            content_div = soup.find('div', {'id': 'mw-content-text'})
            
            # Remove unwanted elements
            if content_div:
                # Remove references, tables, navboxes, etc.
                for element in content_div.select('.reference, .reflist, .navbox, .infobox, .thumb, .mw-editsection'):
                    if element:
                        element.decompose()
            
            # Extract the cleaned content
            wiki_content = content_div.get_text(separator='\n') if content_div else ""
            
            if self.debug:
                log_success(f"Extracted {len(wiki_content)} chars from Wikipedia", "WebScraper")
            
            # Return the same content for all strategies
            return {
                name: ScrapingResult(
                    url=url,
                    success=True,
                    content=wiki_content,
                    metadata={"strategy": name, "source": "wikipedia"}
                ) for name in self.strategy_objects.keys()
            }
            
        except Exception as e:
            # If Wikipedia-specific extraction fails, fall back to regular scraping
            if self.debug:
                log_error(f"Wikipedia extraction failed", "WebScraper", e)
                log_info(f"Falling back to regular extraction", "WebScraper")
            return await super()._fetch_and_extract(url)
            
    def _log_scraping_results(self, results: Dict[str, ScrapingResult], url: str):
        """Log scraping results in a structured way."""
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        if success_count == total_count:
            log_success(f"All methods successful", "WebScraper")
        elif success_count == 0:
            log_error(f"All methods failed", "WebScraper")
        else:
            log_warning(f"{success_count}/{total_count} methods worked", "WebScraper")
            
        # Log the content length for successful strategies
        content_lengths = []
        for name, result in results.items():
            if result.success:
                content_length = len(result.content or "")
                content_lengths.append(content_length)
        
        if content_lengths:
            total_content = sum(content_lengths)
            log_output(f"Extracted {total_content} chars total", "WebScraper")
