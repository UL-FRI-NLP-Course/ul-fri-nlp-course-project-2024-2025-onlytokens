# Set tokenizers parallelism to avoid deadlocks with fork
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#set environ
os.environ["VERBOSE"] = "True"
verbose = os.environ["VERBOSE"] == "True"

import yaml
from typing import Dict, List, Any, Optional
import asyncio
import openai

from rag_search.scraping.quality_scorer import QualityImprover
from rag_search.utils.pipeline_logger import PipelineLogger
from rag_search.processing.llm_query_enhancer import LLMQueryEnhancer
from rag_search.processing.retriever import CosineRetriever, Retriever
from rag_search.processing.reranker import JinaAIReranker, Reranker

# Default configuration as a fallback
CONFIG = {
    "api": {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_url": "https://api.openai.com/v1",
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-2024-08-06"
        },
        "local": {
            "api_key": "sk-or-v1-1234567890",
            "embedding_url": "http://localhost:8000/v1/",
            "llm_url": "http://localhost:8001/v1/",
            "embedding_model": "BAAI/bge-multilingual-gemma2",
            "llm_model": "nemotron"
        }
    },
    "pipeline": {
        "max_sources": 10,     # Maximum number of sources to process in the pipeline
        "debug": False,        # Enable/disable debug mode for detailed logging
        "log_dir": "logs"      # Directory where pipeline logs are stored
    },
    "query_enhancer": {
        "max_queries": 3,      # Maximum number of enhanced queries to generate
        "verbose": verbose        # Enable detailed logging for query enhancement process
    },
    "search_provider": {
        "verbose": verbose,        # Enable detailed logging for search operations
        "instance_url": "http://localhost:5555/search"
    },
    "web_scraper": {
        "strategies": ["lukas"],  # List of scraping strategies to use
        "debug": verbose,            # Enable debug mode for scraping operations
        "filter_content": False   # Whether to filter scraped content
    },
    "quality_improver": {
        "verbose": verbose,          # Enable detailed logging for quality improvement
        "enable_quality_model": False,
        "min_quality_score": 0.2
    },
    "chunker": {
        "verbose": verbose,          # Enable detailed logging for text chunking
        "chunk_size": 1000,       # Size of each text chunk in characters
        "chunk_overlap": 100      # Number of overlapping characters between chunks
    },
    "embedder": {
        "verbose": verbose,          # Enable detailed logging for embedding generation
        "max_tokens": 4096,       # Maximum number of tokens to process at once
        "batch_size": 50          # Number of texts to embed in a single batch
    },
    "retriever": {
        "verbose": verbose,          # Enable detailed logging for retrieval operations
        "top_k": 50               # Number of top candidates to retrieve for reranking
    },
    "reranker": {
        "verbose": verbose,          # Enable detailed logging for reranking process
        "top_k": 10,             # Number of top results to keep after reranking
        "batch_size": 16,         # Number of candidates to rerank in a single batch
        "max_length": 1024        # Maximum text length for reranking
    },
    "context_builder": {
        "verbose": verbose           # Enable detailed logging for context building
    },
    "llm_provider": {
        "verbose": verbose           # Enable detailed logging for LLM operations
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use default CONFIG."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Replace environment variables
            def replace_env_vars(obj):
                if isinstance(obj, dict):
                    return {k: replace_env_vars(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_env_vars(item) for item in obj]
                elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                    env_var = obj[2:-1]
                    return os.getenv(env_var, obj)
                return obj
            
            return replace_env_vars(config)
    return CONFIG

def get_api_config(config: Dict[str, Any], use_openai: bool = False) -> Dict[str, str]:
    """Get API configuration based on whether using OpenAI or local endpoints."""
    api_section = config["api"]["openai"] if use_openai else config["api"]["local"]
    
    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set when use_openai=True")
        return {
            "api_key": api_key,
            "embedding_url": api_section["api_url"],
            "llm_url": api_section["api_url"],
            "embedding_model": api_section["embedding_model"],
            "llm_model": api_section["llm_model"]
        }
    return {
        "api_key": api_section["api_key"],
        "embedding_url": api_section["embedding_url"],
        "llm_url": api_section["llm_url"],
        "embedding_model": api_section["embedding_model"],
        "llm_model": api_section["llm_model"]
    }

from rag_search.llm.openai_provider import OpenAIProvider
from rag_search.search.base import SearchProvider
from rag_search.scraping.crawl4ai_scraper import WebScraper
from rag_search.processing.chunker import Chunker
from rag_search.processing.embedder import Embedder, OpenAIEmbedder, SentenceTransformerEmbedder
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
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
        query_enhancer: Optional[LLMQueryEnhancer] = None,
        max_sources: int = 3,
        debug: bool = False,
        log_dir: str = "logs"
    ):
        # Initialize logger first
        self.logger = PipelineLogger(log_dir=log_dir)
        
        # Pass logger to components that support it
        embedder.pipeline_logger = self.logger
        retriever.pipeline_logger = self.logger
        reranker.pipeline_logger = self.logger
        
        self.search_provider = search_provider
        self.web_scraper = web_scraper
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.query_enhancer = query_enhancer
        self.max_sources = max_sources
        self.debug = debug
        
        # Initialize conversation history and content storage
        self.conversation_history = []
        self.embedded_chunks = None
        self.current_context = None
    
    async def search(self, query: str) -> Dict[str, Any]:
        """Perform search and return raw results."""
        try:
            # Enhance query if a query enhancer is available
            if self.query_enhancer:
                enhanced_queries = await self.query_enhancer.enhance(query)
                # Use all enhanced queries to search
                all_results = []
                for search_query in enhanced_queries.enhanced_queries:
                    results = await self.search_provider.search(search_query, num_results=self.max_sources)
                    all_results.append(results)
                
                # Merge results from all queries
                merged_results = self._merge_search_results(all_results)
                
                self.logger.log("query_enhancement", {
                    "original_query": query,
                    "enhanced_queries": enhanced_queries.enhanced_queries,
                    "num_results": len(merged_results.get('organic', []))
                })
                
                return merged_results
            else:
                results = await self.search_provider.search(query, num_results=self.max_sources)
                self.logger.log("search", {
                    "query": query,
                    "num_results": len(results.get('organic', []))
                })
                return results
        except Exception as e:
            self.logger.log_error("search", e, {"query": query})
            raise
    
    def _merge_search_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple search results into a single result set."""
        merged = {'organic': []}
        seen_urls = set()
        
        # Combine organic results from all searches
        for results in results_list:
            if 'organic' in results:
                for result in results['organic']:
                    url = result.get('link')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        merged['organic'].append(result)
        
        return merged
    
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

            # Embed chunks and store for later use
            self.embedded_chunks = self.embedder.embed_chunks(all_chunks)
            self.logger.log("embedding", {
                "num_chunks": len(all_chunks),
                "embedding_dim": len(self.embedded_chunks[0]['embedding']) if self.embedded_chunks else 0
            })
            
            # Get initial candidates using cosine similarity retrieval
            initial_candidates = self.retriever.retrieve(self.embedded_chunks, query)
            self.logger.log("initial_retrieval", {
                "num_candidates": len(initial_candidates),
                "top_candidate_score": initial_candidates[0]['similarity'] if initial_candidates else None,
                "candidates": [{
                    "content": c["content"],
                    "url": c["url"],
                    "similarity": c["similarity"],
                    "strategy": c["strategy"]
                } for c in initial_candidates]  # Log top 10 for reasonable size
            })
            
            # Extract just the content and metadata for reranking
            candidates_for_reranking = []
            for chunk in initial_candidates:
                candidate = {
                    'content': chunk['content'],
                    'url': chunk['url'],
                    'strategy': chunk['strategy'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks']
                }
                candidates_for_reranking.append(candidate)
            
            # Rerank the candidates using the cross-encoder
            reranked_chunks = self.reranker.rerank(candidates_for_reranking, query)
            self.logger.log("reranking", {
                "num_chunks_after_rerank": len(reranked_chunks),
                "top_chunk_score": reranked_chunks[0]['similarity'] if reranked_chunks else None,
                "reranked_chunks": [{
                    "content": c["content"],
                    "url": c["url"],
                    "similarity": c["similarity"],
                    "strategy": c["strategy"]
                } for c in reranked_chunks]
            })
            
            return reranked_chunks
        except Exception as e:
            self.logger.log_error("process_content", e)
            raise
    
    def build_context(self, processed_content: List[Dict[str, Any]], search_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Build context from processed content and search results."""
        try:
            context = self.context_builder.build(processed_content, search_results, query=query)
            self.logger.log("context_building", {
                "context_length": len(context["context"]) if isinstance(context, dict) else len(context),
                "num_chunks_used": len(processed_content),
                "context": context,
                "chunks_used": [{
                    "content": c["content"],
                    "url": c["url"],
                    "similarity": c.get("similarity"),
                    "strategy": c["strategy"]
                } for c in processed_content]
            })
            
            return context
        except Exception as e:
            self.logger.log_error("build_context", e)
            raise
    
    async def generate_response(self, context: str, query: str, is_followup: bool = False) -> str:
        """Generate response using LLM with context."""
        try:
            # Build messages including conversation history
            messages = []
            
            # Add system message first
            if isinstance(context, dict):
                # New format from LukaContextBuilder
                messages.append({"role": "system", "content": context["system"]})
            else:
                # Legacy format or simple string context
                system_prompt = """You are a helpful AI assistant. Answer the user's questions based on the provided context.
If you cannot find the answer in the context, say so - do not make up information."""
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current context and query
            if isinstance(context, dict):
                messages.append({"role": "user", "content": context["context"]})
            else:
                # Legacy format or simple string context
                combined_message = f"{context}\n\nQuestion: {query}"
                messages.append({"role": "user", "content": combined_message})
            
            response = await self.llm_provider.generate(messages)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            self.logger.log("response_generation", {
                "query": query,
                "context_length": len(context) if isinstance(context, str) else len(context["context"]),
                "response_length": len(response),
                "response": response,
                "conversation_length": len(self.conversation_history),
                "is_followup": is_followup
            })
            return response
        except Exception as e:
            self.logger.log_error("generate_response", e, {
                "query": query,
                "context_length": len(context) if isinstance(context, str) else len(context["context"])
            })
            raise
    
    def clear_conversation(self):
        """Clear the conversation history and cached content."""
        self.conversation_history = []
        self.embedded_chunks = None
        self.current_context = None
        self.logger.log("conversation_cleared", {
            "message": "Conversation history and cached content have been cleared"
        })
    
    async def run(self, query: str) -> str:
        """Execute the complete pipeline."""
        try:
            self.logger.log("pipeline_start", {"query": query})
            
            # Check if this is a follow-up query
            is_followup = len(self.conversation_history) > 0
            
            if not is_followup:
                # Only perform search and scraping for the first query
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

                # Process content - this will store embedded chunks
                processed_content = await self.process_content(scraped_content, query)
                
                # Build context
                self.current_context = self.build_context(processed_content, search_results, query)
                
            else:
                # For follow-up queries, use stored embedded chunks
                if not self.embedded_chunks:
                    raise ValueError("No embedded chunks available for follow-up query")
                
                # Get initial candidates using cosine similarity retrieval
                initial_candidates = self.retriever.retrieve(self.embedded_chunks, query)
                self.logger.log("followup_retrieval", {
                    "num_candidates": len(initial_candidates),
                    "top_candidate_score": initial_candidates[0]['similarity'] if initial_candidates else None
                })
                
                # Extract just the content and metadata for reranking
                candidates_for_reranking = []
                for chunk in initial_candidates:
                    candidate = {
                        'content': chunk['content'],
                        'url': chunk['url'],
                        'strategy': chunk['strategy'],
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk['total_chunks']
                    }
                    candidates_for_reranking.append(candidate)
                
                # Rerank the retrieved candidates
                reranked_chunks = self.reranker.rerank(candidates_for_reranking, query)
                self.logger.log("followup_reranking", {
                    "num_reranked": len(reranked_chunks),
                    "top_reranked_score": reranked_chunks[0]['similarity'] if reranked_chunks else None
                })
                
                # Build new context from reranked chunks
                self.current_context = self.build_context(reranked_chunks, {}, query)
            
            # Generate response
            response = await self.generate_response(self.current_context, query, is_followup)
            
            self.logger.log("pipeline_complete", {
                "query": query,
                "is_followup": is_followup,
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
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Search Pipeline')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI API instead of local endpoints')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Get unified API configuration
    api_config = get_api_config(config, args.use_openai)

    # Initialize OpenAI clients using unified configuration
    embeding_client = openai.OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["embedding_url"]
    )
    llm_client = openai.OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["llm_url"]
    )

    query_enhancer = LLMQueryEnhancer(
        openai_client=llm_client,
        model=api_config["llm_model"],
        max_queries=config["query_enhancer"]["max_queries"],
        verbose=config["query_enhancer"]["verbose"]
    )
    
    search_provider = SearXNGProvider(
        verbose=config["search_provider"]["verbose"],
        instance_url=config["search_provider"]["instance_url"]
    )
    quality_improver = QualityImprover(verbose=config["quality_improver"]["verbose"])
    web_scraper = WebScraper(
        strategies=config["web_scraper"]["strategies"],
        debug=config["web_scraper"]["debug"], 
        llm_base_url=api_config["llm_url"],
        user_query=None, 
        quality_improver=quality_improver,
        min_quality_score=config["quality_improver"]["min_quality_score"],
        enable_quality_model=config["quality_improver"]["enable_quality_model"],
    )

    chunker = Chunker(
        verbose=config["chunker"]["verbose"],
        chunk_size=config["chunker"]["chunk_size"],
        chunk_overlap=config["chunker"]["chunk_overlap"]
    )

    # Initialize embedder for initial retrieval
    embedder = OpenAIEmbedder(
        openai_client=embeding_client,
        model_name=api_config["embedding_model"],
        verbose=config["embedder"]["verbose"],
        max_tokens=config["embedder"]["max_tokens"],
        batch_size=config["embedder"]["batch_size"]
    )
    
    # Initialize retriever for first stage
    retriever = CosineRetriever(
        embedder=embedder, 
        verbose=config["retriever"]["verbose"],
        top_k=config["retriever"]["top_k"]
    )
    
    # Initialize reranker for second stage
    reranker = JinaAIReranker(
        verbose=config["reranker"]["verbose"],
        top_k=config["reranker"]["top_k"],
        batch_size=config["reranker"]["batch_size"],
        max_length=config["reranker"]["max_length"]
    )
    
    context_builder = LukaContextBuilder(verbose=config["context_builder"]["verbose"])
    llm_provider = OpenAIProvider(
        model=api_config["llm_model"],
        api_key=api_config["api_key"],
        api_base=api_config["llm_url"],
        verbose=config["llm_provider"]["verbose"]
    )

    # Initialize pipeline with both retriever and reranker
    pipeline = RAGSearchPipeline(
        search_provider=search_provider,
        web_scraper=web_scraper,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm_provider=llm_provider,
        query_enhancer=query_enhancer,
        max_sources=config["pipeline"]["max_sources"],
        debug=config["pipeline"]["debug"]
    )
    
    # Run pipeline in continuous chat mode
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
    print("\033[94mWelcome! You can start chatting. Type 'exit' to end the conversation, 'clear' to start a new one, or 'new' to force a new search.\033[0m")
    
    while True:
        query = input("\033[94mYou: \033[0m")  # Blue color for input prompt
        
        if query.lower() == 'exit':
            print("\033[94mGoodbye!\033[0m")
            break
        elif query.lower() == 'clear':
            pipeline.clear_conversation()
            print("\033[94mConversation cleared. You can start a new chat.\033[0m")
            continue
        elif query.lower() == 'new':
            # Clear conversation to force a new search
            pipeline.clear_conversation()
            
        try:
            response = pipeline.run_sync(query)
            #print("\033[92mAssistant:\033[0m", response)
        except Exception as e:
            print("\033[91mError:\033[0m", str(e))
            print("\033[94mPlease try again with a different query.\033[0m")

