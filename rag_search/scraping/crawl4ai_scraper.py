"""
Modular web scraping implementation using Crawl4AI.
Supports multiple extraction strategies including LLM, CSS, and XPath.
"""

import asyncio
from typing import Dict, List, Optional
import re  # Import the built-in re module
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, ChunkingStrategy, CrawlerRunConfig, CacheMode, RegexChunking
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from rag_search.scraping.basic_web_scraper import ExtractionConfig
from rag_search.scraping.extraction_result import ExtractionResult, print_extraction_result
from rag_search.scraping.quality_scorer import QualityImprover
from rag_search.scraping.strategy_factory import StrategyFactory
from rag_search.utils.logging import log_error, log_info


class MarkdownChunking(ChunkingStrategy):
    """
    Chunking strategy that splits markdown text into logical sections.
    """
    
    def __init__(self, patterns=None, **kwargs):
        """
        Initialize the MarkdownChunking object.
        
        Args:
            patterns (list): A list of regular expression patterns to split text.
        """
        if patterns is None:
            # Patterns for splitting markdown content into meaningful chunks
            patterns = [
                r"(?=^# .*$)",             # Split on H1 headers
                r"(?=^## .*$)",            # Split on H2 headers
                r"(?=^### .*$)",           # Split on H3 headers
                r"(?=^#### .*$)",          # Split on H4 headers
                r"(?=^##### .*$)",         # Split on H5 headers
                r"(?=^###### .*$)",        # Split on H6 headers
                r"(?=^---$)",              # Split on horizontal rules
                r"(?=^\n\n)",              # Split on double newlines (paragraph breaks)
            ]
        self.patterns = patterns
        self.compiled_pattern = re.compile("|".join(patterns), re.MULTILINE)
    
    def chunk(self, text: str) -> list:
        """
        Split text into chunks based on markdown structure.
        
        Args:
            text (str): The text to split.
            
        Returns:
            list: A list of text chunks.
        """
        if not text or not text.strip():
            return []
            
        # Split the text using the compiled pattern
        chunks = self.compiled_pattern.split(text)
        
        # Combine the first chunk with the second if the first chunk is too small
        if len(chunks) > 1 and len(chunks[0].strip()) < 100:
            chunks[1] = chunks[0] + chunks[1]
            chunks.pop(0)
            
        # Filter out empty chunks and strip whitespace
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Merge small chunks with the next chunk to avoid tiny fragments
        i = 0
        while i < len(chunks) - 1:
            if len(chunks[i]) < 100:  # If chunk is too small
                chunks[i+1] = chunks[i] + "\n\n" + chunks[i+1]
                chunks.pop(i)
            else:
                i += 1
                
        return chunks

class WebScraper:
    """Unified scraper that encapsulates all extraction strategies and configuration"""
    def __init__(
        self, 
        browser_config: Optional[BrowserConfig] = None,
        strategies: List[str] = ['markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction'],
        #TODO: maybe improve promt to extract relevant content to the question
        llm_instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        user_query: Optional[str] = None,
        debug: bool = False,
        enable_quality_model: bool = False,
        llm_base_url: str = "https://localhost:8001/v1/",
        quality_improver: Optional[QualityImprover] = None,
        min_quality_score: float = 0.2
    ):
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=False)
        self.debug = debug
        self.factory = StrategyFactory()
        self.strategies = strategies
        self.llm_instruction = llm_instruction
        self.user_query = user_query
        self.enable_quality_model = enable_quality_model
        self.quality_improver = quality_improver
        self.min_quality_score = min_quality_score
        # Validate strategies
        valid_strategies = {'markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction', 'lukas'}
        invalid_strategies = set(self.strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")
            
        # Initialize strategy map
        self.strategy_map = {
            'markdown_llm': lambda: self.factory.create_llm_strategy('markdown', self.llm_instruction, llm_base_url),
            'html_llm': lambda: self.factory.create_llm_strategy('html', self.llm_instruction, llm_base_url),
            'fit_markdown_llm': lambda: self.factory.create_llm_strategy('fit_markdown', self.llm_instruction, llm_base_url),
            'css': self.factory.create_css_strategy,
            'xpath': self.factory.create_xpath_strategy,
            'no_extraction': self.factory.create_no_extraction_strategy(),
            'lukas': lambda: self.factory.create_lukas_strategy(),
            #TODO what is cosien extracor?
            #'cosine': lambda: self.factory.create_cosine_strategy(debug=self.debug)
        }

    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        #TODO: investiage filter_content but i think pruning is ok one 
        content_filter = PruningContentFilter(user_query=self.user_query) if self.user_query else PruningContentFilter()
        #todo investigae craeler config
        #NOTE content filer to bea pplied on html before conversion to markdown

        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=content_filter
            ),
            chunking_strategy=MarkdownChunking() #nOTE how to solit generated markdown? in chunks
        )

    async def scrape(self, url: str) -> Dict[str, ExtractionResult]:
        """
        Scrape URL using configured strategies
        
        Args:
            url: Target URL to scrape
        """
        # Handle Wikipedia URLs
        if 'wikipedia.org/wiki/' in url:
            from rag_search.scraping.utils import get_wikipedia_content
            try:
                content = get_wikipedia_content(url)
                # Create same result for all strategies since we're using Wikipedia content
                return {
                    strategy_name: ExtractionResult(
                        name=strategy_name,
                        success=True,
                        content=content
                    ) for strategy_name in self.strategies
                }
            except Exception as e:
                if self.debug:
                    print(f"Debug: Wikipedia extraction failed: {str(e)}")
                # If Wikipedia extraction fails, fall through to normal scraping
        
        # Normal scraping for non-Wikipedia URLs or if Wikipedia extraction failed
        results = {}
        for strategy_name in self.strategies:
            config = ExtractionConfig(
                name=strategy_name,
                strategy=self.strategy_map[strategy_name]()
            )
            result = await self.extract(config, url)
            results[strategy_name] = result
            
        return results
    
    async def scrape_many(self, urls: List[str]) -> Dict[str, Dict[str, ExtractionResult]]:
        """
        Scrape multiple URLs using configured strategies in parallel
        
        Args:
            urls: List of target URLs to scrape
            
        Returns:
            Dictionary mapping URLs to their extraction results
        """
        # Create tasks for all URLs
        tasks = [self.scrape(url) for url in urls]
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Build results dictionary
        results = {}
        for url, result in zip(urls, results_list):
            results[url] = result
            
        return results

    async def extract(self, extraction_config: ExtractionConfig, url: str) -> ExtractionResult:
        """Internal method to perform extraction using specified strategy"""
        try:
            config = self._create_crawler_config()
            config.extraction_strategy = extraction_config.strategy

            if self.debug:
                log_info(f"Scraping {url} with strategy {extraction_config.name}", "WebScraper")


            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                if isinstance(url, list):
                    #TODO MAKE USE IF THIS  INTEAD OF ONLY ONE URL AT A TIME
                    result = await crawler.arun_many(urls=url, config=config)
                else:
                    result = await crawler.arun(url=url, config=config)

            if self.debug:
                # print(f"Debug: Raw result attributes: {dir(result)}")
                # print(f"Debug: Raw result: {result.__dict__}")
                pass

            # Handle different result formats based on strategy
            content = None
            if result.success:
                if extraction_config.name in ['no_extraction', 'cosine', 'lukas']:
                    if hasattr(result, 'extracted_content') and result.extracted_content:
                        if isinstance(result.extracted_content, list):
                            # Handle list of dictionaries
                            content = '\n\n'.join(item.get('content', '') for item in result.extracted_content)
                        elif isinstance(result.extracted_content, str):
                            # Check if the string is JSON
                            try:
                                # Try to parse as JSON
                                json_content = json.loads(result.extracted_content)
                                if isinstance(json_content, list):
                                    # Handle JSON list of dictionaries
                                    content = '\n\n'.join(item.get('content', '') for item in json_content)
                                else:
                                    # Handle other JSON structures
                                    content = str(json_content)
                            except json.JSONDecodeError:
                                # Not JSON, use the string as is
                                content = result.extracted_content
                        else:
                            content = str(result.extracted_content)

                        # Save to file for debugging
                        # if self.debug:
                        #     with open("source_content_before_quality_model.md", "w") as f:
                        #         f.write(content)
                    elif hasattr(result, 'markdown'):
                        content = result.markdown.raw_markdown
                    elif hasattr(result, 'raw_html'):
                        content = result.raw_html
                    
                    if self.enable_quality_model and content:
                        content = self.quality_improver.filter_quality_content(content, min_quality_score=self.min_quality_score)
                else:
                    content = result.extracted_content
                    if self.enable_quality_model and content:
                        content = self.quality_improver.filter_quality_content(content, min_quality_score=self.min_quality_score)

            if self.debug:
                if content:
                    log_info(f"Scraped content: {content[:100].strip().replace('\n', ' ').replace('\r', ' ')}", "WebScraper")
                else:
                    log_error(f"Unable to scrape content from {url}", "WebScraper")
                # #save to file
                # if content:
                #     with open("source_content.md", "w") as f:
                #         f.write(content)
            

            extraction_result = ExtractionResult(
                name=extraction_config.name,
                success=result.success,
                content=content,
                error=getattr(result, 'error', None)  # Capture error if available
            )
            
            if result.success:
                extraction_result.raw_markdown_length = len(result.markdown.raw_markdown)
                extraction_result.citations_markdown_length = len(result.markdown.markdown_with_citations)
            elif self.debug:
                print(f"Debug: Final extraction result: {extraction_result.__dict__}")

            return extraction_result

        except Exception as e:
            if self.debug:
                import traceback
                print(f"Debug: Exception occurred during extraction:")
                print(traceback.format_exc())
            
            return ExtractionResult(
                name=extraction_config.name,
                success=False,
                error=str(e)
            )

async def main():
    # Example usage with single URL
    single_url = "https://example.com/product-page"
    scraper = WebScraper(debug=True, llm_base_url="http://localhost:8001/v1/", enable_quality_model=False)
    results = await scraper.scrape(single_url)
    
    # Print single URL results
    for result in results.values():
        print_extraction_result(result)

    # Example usage with multiple URLs
    urls = [
        "https://example.com",
        "https://python.org",
        "https://github.com"
    ]
    
    multi_results = await scraper.scrape_many(urls)
    
    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extraction_result(result)

if __name__ == "__main__":
    asyncio.run(main())
