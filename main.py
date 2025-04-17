from typing import Dict, List, Any, Optional
import asyncio
import os

import openai

from rag_search.scraping.quality_scorer import QualityImprover
from rag_search.utils.pipeline_logger import PipelineLogger

# Set tokenizers parallelism to avoid deadlocks with fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_search.llm.openai_provider import OpenAIProvider
from rag_search.search.base import SearchProvider
from rag_search.scraping.crawl4ai_scraper import WebScraper
from rag_search.processing.chunker import Chunker
from rag_search.processing.embedder import Embedder, OpenAIEmbedder, SentenceTransformerEmbedder
from rag_search.processing.reranker import CosineReranker, Reranker
from rag_search.processing.query_enhancer import QueryEnhancer
from rag_search.context.builder import ContextBuilder, LukaContextBuilder
from rag_search.llm.provider import LLMProvider
from rag_search.search.searxng import SearXNGProvider
from rag_search.context.builder import SimpleContextBuilder
from rag_search.utils.logging import log_info

import litellm

class RAGSearchPipeline:
    """Main pipeline for RAG search, orchestrating the entire process."""
    
    def __init__(
        self,
        search_provider: SearXNGProvider,
        web_scraper: WebScraper,
        chunker: Chunker,
        embedder: Embedder,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
        query_enhancer: Optional[QueryEnhancer] = None,
        max_sources: int = 3,
        debug: bool = False,
        log_dir: str = "logs"
    ):
        # Initialize logger first
        self.logger = PipelineLogger(log_dir=log_dir)
        
        # Pass logger to components that support it
        embedder.pipeline_logger = self.logger
        reranker.pipeline_logger = self.logger
        
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
        try:
            # Enhance query if a query enhancer is available
            if self.query_enhancer:
                enhanced_query = self.query_enhancer.enhance(query)
                search_query = enhanced_query.enhanced_query
                self.logger.log("query_enhancement", {
                    "original_query": query,
                    "enhanced_query": search_query
                })
            else:
                search_query = query
            
            results = await self.search_provider.search(search_query, num_results=self.max_sources)
            self.logger.log("search", {
                "query": search_query,
                "num_results": len(results.get('organic', [])),
                "results": results
            })
            return results
        except Exception as e:
            self.logger.log_error("search", e, {"query": query})
            raise
    
    async def process_content(self, scraped_content: Dict[str, Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Process scraped content for relevance."""
        try:
            all_chunks = []
            debug_save = {}
            
            self.logger.log("content_processing_start", {
                "num_urls": len(scraped_content),
                "query": query
            })
            
            # Process each URL and its extraction results
            for url, strategies in scraped_content.items():
                if not isinstance(strategies, dict):
                    continue

                # Create URL section in debug_save
                if url not in debug_save:
                    debug_save[url] = {}
                
                # Extract content from all available strategies
                for strategy_name, extraction_result in strategies.items():
                    # Save all extraction results to debug_save
                    if hasattr(extraction_result, 'content'):
                        debug_save[url][strategy_name] = {
                            'content': extraction_result.content if extraction_result.content else "No content",
                            'success': getattr(extraction_result, 'success', False)
                        }
                    else:
                        debug_save[url][strategy_name] = {
                            'content': "No content attribute",
                            'success': getattr(extraction_result, 'success', False)
                        }
                    
                    # Only process successful extractions with content for RAG
                    if (hasattr(extraction_result, 'success') and 
                        extraction_result.success and 
                        hasattr(extraction_result, 'content') and 
                        extraction_result.content):

                        # Simply split the text content
                        text_chunks = self.chunker.split_text(extraction_result.content)
                        
                        # Create chunk objects with source metadata
                        for i, chunk_text in enumerate(text_chunks):
                            chunk = {
                                'content': chunk_text,
                                'url': url,
                                'strategy': strategy_name,
                                'chunk_index': i,
                                'total_chunks': len(text_chunks)
                            }
                            all_chunks.append(chunk)

                self.logger.log("url_processing", {
                    "url": url,
                    "num_strategies": len(strategies),
                    "successful_strategies": [
                        strategy for strategy, result in strategies.items()
                        if getattr(result, 'success', False)
                    ]
                })

            # Save to nice markdown file
            with open("debug_save.md", "w") as f:
                for url, strategies in debug_save.items():
                    f.write(f"# URL: {url}\n\n")
                    for strategy_name, result in strategies.items():
                        f.write(f"## Strategy: {strategy_name} (Success: {result['success']})\n\n")
                        f.write(result['content'])
                        f.write("\n\n---\n\n")

            # Generate embeddings
            embedded_chunks = self.embedder.embed_chunks(all_chunks)
            self.logger.log("embedding", {
                "num_chunks": len(all_chunks),
                "embedding_dim": len(embedded_chunks[0]['embedding']) if embedded_chunks else 0
            })
            
            # Rerank chunks
            reranked_chunks = self.reranker.rerank(embedded_chunks, query)
            self.logger.log("reranking", {
                "num_chunks_after_rerank": len(reranked_chunks),
                "top_chunk_score": reranked_chunks[0]['similarity'] if reranked_chunks else None
            })
            
            return reranked_chunks
        except Exception as e:
            self.logger.log_error("process_content", e)
            raise
    
    def build_context(self, processed_content: List[Dict[str, Any]], search_results: Dict[str, Any]) -> str:
        """Build context from processed content and search results."""
        try:
            context = self.context_builder.build(processed_content, search_results)
            self.logger.log("context_building", {
                "context_length": len(context),
                "num_chunks_used": len(processed_content)
            })
            return context
        except Exception as e:
            self.logger.log_error("build_context", e)
            raise
    
    async def generate_response(self, context: str, query: str) -> str:
        """Generate response using LLM with context."""
        try:
            input_prompt = f"""
            Context: {context}
            Question: {query}
            """

            messages = [
                {"role": "user", "content": input_prompt}
            ]
            
            response = await self.llm_provider.generate(messages)
            self.logger.log("response_generation", {
                "query": query,
                "context_length": len(context),
                "response_length": len(response)
            })
            return response
        except Exception as e:
            self.logger.log_error("generate_response", e, {
                "query": query,
                "context_length": len(context)
            })
            raise
    
    async def run(self, query: str) -> str:
        """Execute the complete pipeline."""
        try:
            self.logger.log("pipeline_start", {"query": query})
            
            # Search
            search_results = await self.search(query)
                
            # Extract URLs from search results
            urls = self._extract_urls(search_results)
            urls = urls[:self.max_sources]
            
            self.logger.log("url_extraction", {
                "num_urls": len(urls),
                "urls": urls
            })
            
            # Scrape content from URLs
            scraped_content = await self.web_scraper.scrape_many(urls)
            self.logger.log("scraping", {
                "num_urls_scraped": len(scraped_content)
            })

            # Process
            processed_content = await self.process_content(scraped_content, query)
            
            # Build context
            context = self.build_context(processed_content, search_results)
            
            # Generate response
            response = await self.generate_response(context, query)
            
            self.logger.log("pipeline_complete", {
                "query": query,
                "total_urls": len(urls),
                "response_length": len(response)
            })
                
            return response
        except Exception as e:
            self.logger.log_error("pipeline_run", e, {"query": query})
            raise
    
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

    embeding_client = openai.OpenAI(api_key="sk-or-v1-1234567890",base_url="http://localhost:8000/v1/")

    query_enhancer = QueryEnhancer(verbose=True)
    search_provider = SearXNGProvider(verbose=True)
    quality_improver = QualityImprover(verbose=True)
    web_scraper = WebScraper(strategies=['lukas'],debug=True, llm_base_url="http://localhost:8001/v1/", filter_content=False,user_query=None, quality_improver=quality_improver)

    chunker = Chunker(verbose=True,chunk_size=1000,chunk_overlap=100)

    embedder = OpenAIEmbedder(
        openai_client=embeding_client,
        model_name="BAAI/bge-multilingual-gemma2",
        verbose=True,
        max_tokens=4096,  # Set max tokens to avoid overflow
        batch_size=50  # Process in smaller batches
    )
    reranker = CosineReranker(embedder=embedder, verbose=False,top_k=10)
    context_builder = LukaContextBuilder(verbose=True)
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
        max_sources=10,
        debug=False
    )
    
    # Run pipeline
    print("\033[92m")  # Green color
    print(r"""
    ____        __     ______      __                
   / __ \____  / /_  _/_  __/___  / /_____  ____  ___
  / / / / __ \/ / / / // / / __ \/ //_/ _ \/ __ \/ __/
 / /_/ / / / / / /_/ // / / /_/ / ,< /  __/ / / /\ \  
/_____/_/ /_/_/\__, //_/  \____/_/|_|\___/_/ /_/___/  
              /____/                                  
    """)
    print("\033[0m")  # Reset color
    query = input("\033[94mWhat do you want to know? \033[0m")  # Blue color for input prompt
    response = pipeline.run_sync(query)

