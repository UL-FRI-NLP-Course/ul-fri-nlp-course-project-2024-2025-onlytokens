from typing import Dict, List, Any, Optional
import asyncio
import os

# Set tokenizers parallelism to avoid deadlocks with fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_search.llm.openai_provider import OpenAIProvider
from rag_search.search.base import SearchProvider
from rag_search.scraping.scraper import WebScraper
from rag_search.processing.chunker import Chunker
from rag_search.processing.embedder import Embedder, SentenceTransformerEmbedder
from rag_search.processing.reranker import CosineReranker, Reranker
from rag_search.processing.query_enhancer import QueryEnhancer
from rag_search.context.builder import ContextBuilder
from rag_search.llm.provider import LLMProvider
from rag_search.search.searxng import SearXNGProvider
from rag_search.context.builder import SimpleContextBuilder
from rag_search.utils.logging import log_info


class RAGSearchPipeline:
    """Main pipeline for RAG search, orchestrating the entire process."""
    
    def __init__(
        self,
        search_provider: SearchProvider,
        web_scraper: WebScraper,
        chunker: Chunker,
        embedder: Embedder,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
        query_enhancer: Optional[QueryEnhancer] = None,
        max_sources: int = 3,
        debug: bool = False
    ):
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.chunker = chunker
        self.embedder = embedder
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.query_enhancer = query_enhancer
        self.max_sources = max_sources
        self.debug = debug
    
    async def search(self, query: str) -> Dict[str, Any]:
        """Perform search and return raw results."""
        # Enhance query if a query enhancer is available
        if self.query_enhancer:
            enhanced_query = self.query_enhancer.enhance(query)
            search_query = enhanced_query.enhanced_query
            search_query = query
            
        return await self.search_provider.search(search_query, num_results=self.max_sources)
    
    async def scrape_content(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract content from search results URLs."""
        # Extract URLs from search results
        urls = self._extract_urls(search_results)
        
        # Limit to max_sources
        urls = urls[:self.max_sources]
        
        # Scrape content from URLs
        return await self.web_scraper.scrape_many(urls)
    
    async def process_content(self, scraped_content: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Process scraped content for relevance."""
        
        #e
        # Chunk content
        chunks = self.chunker.split_scraped_content(scraped_content)
        
        # Generate embeddings
        embedded_chunks = self.embedder.embed_chunks(chunks)
        
        # Rerank chunks by relevance to query
        return self.reranker.rerank(embedded_chunks, query)
    
    def build_context(self, processed_content: List[Dict[str, Any]], search_results: Dict[str, Any]) -> str:
        """Build context from processed content and search results."""
        return self.context_builder.build(processed_content, search_results)
    
    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using LLM with context."""

        input_prompt = f"""
        Context: {context}
        Question: {query}
        """

        messages = [
            {"role": "user", "content": input_prompt}
        ]
        return await self.llm_provider.generate(messages)
    
    async def run(self, query: str) -> str:
        """Execute the complete pipeline."""
        # Search
        search_results = await self.search(query)
            
        # Scrape
        scraped_content = await self.scrape_content(search_results)
       
        # Process
        processed_content = await self.process_content(scraped_content, query)
        
        # Build context
        context = self.build_context(processed_content, search_results)
        
        # Generate response
        response = await self.generate_response(context, query)
            
        return response
    
    def run_sync(self, query: str) -> str:
        """Synchronous version of run."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run(query))
    
    def _extract_urls(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract URLs from search results, prioritizing the latest information."""
        urls = []
        
        # Extract from organic results and sort by date
        organic_results = search_results.get('organic', [])
        
        # Filter results with date information and sort them (latest first)
        dated_results = [result for result in organic_results if result.get('date')]
        dated_results.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Add results without date information at the end
        undated_results = [result for result in organic_results if not result.get('date')]
        
        # Combine sorted results
        sorted_results = dated_results + undated_results

        log_info(f"Found {len(sorted_results)} organic results", "RAGSearchPipeline")
        for result in sorted_results:
            log_info(f"Result: {result['title']} - {result['date']} - {result['link']}", "RAGSearchPipeline")
        
        # Extract URLs from sorted results
        for result in sorted_results:
            if 'link' in result:
                urls.append(result['link'])
                
        return urls


if __name__ == "__main__":
    # Initialize components with verbose logging enabled
    query_enhancer = QueryEnhancer(verbose=True)
    search_provider = SearXNGProvider(verbose=True)
    web_scraper = WebScraper(debug=True)
    chunker = Chunker(verbose=True)
    embedder = SentenceTransformerEmbedder(device="mps", verbose=True)
    reranker = CosineReranker(embedder=embedder, verbose=False)
    context_builder = SimpleContextBuilder(verbose=True)
    llm_provider = OpenAIProvider(
        model="nemotron",
        api_key="sk-or-v1-1234567890",
        api_base="http://localhost:8001/v1/",
        verbose=True
    )

    # Initialize pipeline
    pipeline = RAGSearchPipeline(
        search_provider=search_provider,
        web_scraper=web_scraper,
        chunker=chunker,
        embedder=embedder,
        reranker=reranker,
        context_builder=context_builder,
        llm_provider=llm_provider,
        query_enhancer=query_enhancer,
        max_sources=100,
        debug=False
    )
    
    # Run pipeline
    query = "Kdo je Anze Jensterle (CraftByte)"
    response = pipeline.run_sync(query)

