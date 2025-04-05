#!/usr/bin/env python3
"""
Example of using LiteLLMProvider to connect to different LLM API endpoints.
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv

from rag_search.llm import LiteLLMProvider

async def main():
    """Run the example"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example using LiteLLMProvider")
    parser.add_argument("--model_url", required=True, help="Model URL or identifier")
    parser.add_argument("--api_key", help="API key (if not set in env vars)")
    parser.add_argument("--query", default="What is the capital of France?", help="Query to test")
    parser.add_argument("--context", default="France is a country in Europe with Paris as its capital.", 
                       help="Context to provide")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Use API key from args or environment variable
    api_key = args.api_key or os.getenv("API_KEY")
    
    # Initialize the LiteLLM provider
    llm = LiteLLMProvider(
        model_url=args.model_url,
        api_key=api_key,
        verbose=args.verbose
    )
    
    print(f"Using model: {args.model_url}")
    print(f"Query: {args.query}")
    print(f"Context: {args.context}")
    
    # Generate a response
    try:
        response = await llm.generate(
            context=args.context,
            query=args.query,
            temperature=0.7,
            max_tokens=100
        )
        
        print("\nResponse:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 