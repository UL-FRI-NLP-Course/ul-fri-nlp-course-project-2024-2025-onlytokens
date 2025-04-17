"""
title: Web Search using RAG Pipeline API
author: OnlyTokens
author_url: https://github.com/EntropyYue/web_search
git_url: https://github.com/EntropyYue/web_search.git
description: This tool searches the web using a RAG Pipeline API and retrieves relevant content
required_open_webui_version: 0.4.0
requirements: requests>=2.31.0,aiohttp>=3.9.0,beautifulsoup4>=4.12.0
version: 0.4.4
license: MIT
"""

import json
import requests
import asyncio
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from urllib.parse import urljoin
from datetime import datetime

class Tools:
    class Valves(BaseModel):
        RAG_API_BASE_URL: str = Field(
            default="http://localhost:8222",
            description="Base URL for the RAG Pipeline API"
        )
        MAX_RESULTS: int = Field(
            default=5,
            description="Maximum number of search results to return"
        )
        REQUEST_TIMEOUT: int = Field(
            default=120,
            description="Timeout for API requests in seconds"
        )

    class UserValves(BaseModel):
        SHOW_FULL_CONTENT: bool = Field(
            default=False,
            description="Show full content of search results instead of snippets"
        )

    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        # Disable built-in citations as we'll handle them manually
        self.citation = False

    async def search_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[Dict], Any]] = None,
        __user__: Dict = None
    ) -> str:
        """
        Search the web and retrieve content from relevant pages.

        :param query: The search query to use
        :param __event_emitter__: Event emitter for status updates and citations
        :param __user__: User information including valves
        :return: The formatted search results as text
        """
        user_valves = __user__.get("valves") if __user__ else None
        show_full_content = user_valves.SHOW_FULL_CONTENT if user_valves else False

        try:
            # Initial status - Starting search
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Starting search for: {query}",
                            "done": False,
                            "hidden": False
                        }
                    }
                )
            
            # Send search request to the API
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Querying search engine...",
                            "done": False,
                            "hidden": False
                        }
                    }
                )
            
            # Prepare and send the request
            search_endpoint = urljoin(self.valves.RAG_API_BASE_URL, "/search")
            response = requests.post(
                search_endpoint,
                json={"query": query},
                headers=self.headers,
                timeout=self.valves.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = data.get("results", [])
            if len(results) > self.valves.MAX_RESULTS:
                results = results[:self.valves.MAX_RESULTS]
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Processing {len(results)} search results...",
                            "done": False,
                            "hidden": False
                        }
                    }
                )
            
            # Format response based on user preference
            formatted_results = []
            for result in results:
                # Extract content from the result structure
                content = ""
                if isinstance(result, dict):
                    if "content" in result:
                        content = result["content"]
                    elif "no_extraction" in result:
                        content = result["no_extraction"].get("content", "")
                    else:
                        for strategy_name, strategy_result in result.items():
                            if isinstance(strategy_result, dict) and strategy_result.get("success"):
                                content = strategy_result.get("content", "")
                                break
                
                # Get or generate snippet
                snippet = result.get("snippet", "")
                if not snippet and content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                
                formatted_result = {
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", result.get("link", "")),
                    "content": content,
                    "snippet": snippet
                }
                
                formatted_results.append(formatted_result)

            # Emit citations for each result
            if __event_emitter__:
                for result in formatted_results:
                    url = result["url"]
                    title = result["title"]
                    text = result["content"] if show_full_content else result["snippet"]
                    if url and text:
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [text],
                                    "metadata": [
                                        {
                                            "date_accessed": datetime.now().isoformat(),
                                            "source": url
                                        }
                                    ],
                                    "source": {
                                        "name": title,
                                        "url": url
                                    }
                                }
                            }
                        )

            # Format final response text
            response_text = []
            for i, result in enumerate(formatted_results, 1):
                title = result["title"]
                url = result["url"]
                text = result["content"] if show_full_content else result["snippet"]
                
                response_text.append(f"{i}. {title}")
                response_text.append(f"   URL: {url}")
                response_text.append(f"   {text}")
                response_text.append("")
            
            # Add sources section
            response_text.append("Sources:")
            for result in formatted_results:
                if url := result["url"]:
                    response_text.append(f"- {url}")
            
            # Complete status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Search completed with {len(results)} results",
                            "done": True,
                            "hidden": True  # Hide the status after completion
                        }
                    }
                )
            
            return "\n".join(response_text)
            
        except Exception as e:
            error_message = str(e)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error during search: {error_message}",
                            "done": True,
                            "hidden": False
                        }
                    }
                )
            return f"Error occurred during search: {error_message}"
    
    async def get_website(
        self, 
        url: str, 
        __event_emitter__=None,
        __user__: Dict = None
    ) -> str:
        """
        Retrieve content from a specific website.

        :param url: The URL to retrieve content from
        :return: The content of the website as text.
        """
        # Initial status update
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Retrieving content from: {url}",
                        "done": False
                    }
                }
            )
        
        try:
            # Request the website content via the API
            website_endpoint = urljoin(self.valves.RAG_API_BASE_URL, "/get_website")
            
            response = requests.post(
                website_endpoint,
                json={"url": url},
                headers=self.headers,
                timeout=self.valves.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Ensure data is in the expected format
            if not isinstance(data, list) and not isinstance(data, dict):
                return f"Error: Unexpected response format from API for URL: {url}"
            
            # Convert to list if it's a dictionary
            if isinstance(data, dict):
                data = [data]
            elif not data:
                data = []
            
            # Add citation
            if self.citation and __event_emitter__:
                for item in data:
                    if "title" in item and "content" in item:
                        title = item["title"]
                        content = item["content"]
                        
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [content],
                                    "metadata": [
                                        {
                                            "date_accessed": datetime.now().isoformat(),
                                            "source": url
                                        }
                                    ],
                                    "source": {"name": title, "url": url}
                                }
                            }
                        )
            
            # Completion status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Website content successfully retrieved",
                            "done": True
                        }
                    }
                )
            
            # Format result for the LLM as text
            result_text = []
            for item in data:
                title = item.get("title", "Untitled")
                content = item.get("content", "No content available")
                
                result_text.append(f"Title: {title}")
                result_text.append(f"URL: {url}")
                result_text.append("\nContent:")
                result_text.append(content)
            
            return "\n\n".join(result_text)
            
        except requests.exceptions.RequestException as e:
            error_message = str(e)
            
            # Send error status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error retrieving website: {error_message}",
                            "done": True
                        }
                    }
                )
            
            return f"Error retrieving content from {url}: {error_message}"

# Example usage
if __name__ == "__main__":
    async def print_event(event):
        """Print events to console"""
        print(f"EVENT: {json.dumps(event, indent=2)}")
    
    async def test_search():
        tool = Tools()
        query = input("Enter a search query: ")
        result = await tool.search_web(query, print_event, {"valves": tool.UserValves()})
        print(f"\nSearch results:\n{result}")
    
    # Run the test
    asyncio.run(test_search()) 