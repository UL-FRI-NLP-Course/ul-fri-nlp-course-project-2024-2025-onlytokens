"""
Simple example using the RAG search pipeline.
"""
import asyncio
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from rag_search.main import RAGSearchPipeline
from rag_search.search.searxng import SearXNGProvider
from rag_search.scraping.scraper import WebScraper
from rag_search.processing.chunker import Chunker
from rag_search.processing.embedder import SentenceTransformerEmbedder
from rag_search.processing.reranker import CosineReranker
from rag_search.context.builder import SimpleContextBuilder
from rag_search.llm.local import LocalLLMProvider

# Default SearXNG instance
DEFAULT_SEARXNG_INSTANCE = "https://searx.thegpm.org"

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Search Example")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--model", type=str, default="", help="Path to local LLM model file")
    parser.add_argument("--searxng", type=str, default=DEFAULT_SEARXNG_INSTANCE, help="SearXNG instance URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Check if model path is provided
    if not args.model:
        print("Error: Please provide a path to a local LLM model file using --model")
        return
    
    # Create pipeline components
    print(f"Initializing search provider...")
    search_provider = SearXNGProvider(
        instance_url=args.searxng
    )
    
    print(f"Initializing web scraper...")
    web_scraper = WebScraper(
        strategies=["no_extraction"],
        debug=args.debug
    )
    
    print(f"Initializing chunker...")
    chunker = Chunker(
        chunk_size=512,
        chunk_overlap=50
    )
    
    print(f"Initializing embedder...")
    embedder = SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2"
    )
    
    print(f"Initializing reranker...")
    reranker = CosineReranker(
        embedder=embedder,
        top_k=5
    )
    
    print(f"Initializing context builder...")
    context_builder = SimpleContextBuilder()
    
    print(f"Initializing LLM provider...")
    llm_provider = LocalLLMProvider(
        model_path=args.model,
        verbose=args.debug
    )
    
    # Create the pipeline
    print(f"Creating RAG search pipeline...")
    pipeline = RAGSearchPipeline(
        search_provider=search_provider,
        web_scraper=web_scraper,
        chunker=chunker,
        embedder=embedder,
        reranker=reranker,
        context_builder=context_builder,
        llm_provider=llm_provider,
        debug=args.debug
    )
    
    # Execute the pipeline
    print(f"Executing search for: {args.query}")
    print("-" * 50)
    
    response = await pipeline.run(args.query)
    
    print("\nRESPONSE:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
