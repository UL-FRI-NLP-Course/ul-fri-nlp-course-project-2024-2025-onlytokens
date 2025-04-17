import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import uuid
from sse_starlette.sse import EventSourceResponse

from main import RAGSearchPipeline
from rag_search.llm.openai_provider import OpenAIProvider
from rag_search.search.searxng import SearXNGProvider
from rag_search.scraping.scraper import WebScraper
from rag_search.processing.chunker import Chunker
from rag_search.processing.embedder import SentenceTransformerEmbedder
from rag_search.processing.reranker import CosineReranker
from rag_search.processing.query_enhancer import QueryEnhancer
from rag_search.context.builder import SimpleContextBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="RAGSearch API", version="1.0.0")

# Models for API requests and responses
class SearchQuery(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    snippet: Optional[str] = None
    score: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    
# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sse_connections: Dict[str, asyncio.Queue] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connection established for client: {client_id}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket connection closed for client: {client_id}")
            
    async def register_sse(self, client_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.sse_connections[client_id] = queue
        logger.info(f"SSE connection registered for client: {client_id}")
        return queue
    
    def unregister_sse(self, client_id: str):
        if client_id in self.sse_connections:
            del self.sse_connections[client_id]
            logger.info(f"SSE connection unregistered for client: {client_id}")
            
    async def send_update(self, client_id: str, data: Dict[str, Any]):
        try:
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_json(data)
            if client_id in self.sse_connections:
                await self.sse_connections[client_id].put(data)
        except Exception as e:
            logger.error(f"Error sending update to client {client_id}: {str(e)}")

manager = ConnectionManager()

# Create a default RAGSearchPipeline
def create_default_pipeline() -> RAGSearchPipeline:
    """Create a default RAGSearchPipeline instance."""
    try:
        query_enhancer = QueryEnhancer()
        search_provider = SearXNGProvider()
        web_scraper = WebScraper()
        chunker = Chunker()
        embedder = SentenceTransformerEmbedder()
        reranker = CosineReranker(embedder=embedder)
        context_builder = SimpleContextBuilder()
        llm_provider = OpenAIProvider(
            model="nemotron",
            api_key="sk-dummy",  # Replace with actual API key in production
            api_base="http://localhost:8001/v1/",
        )
        
        logger.info("Default RAGSearchPipeline created successfully")
        return RAGSearchPipeline(
            search_provider=search_provider,
            web_scraper=web_scraper,
            chunker=chunker,
            embedder=embedder,
            reranker=reranker,
            context_builder=context_builder,
            llm_provider=llm_provider,
            query_enhancer=query_enhancer,
            max_sources=3,
        )
    except Exception as e:
        logger.error(f"Failed to create default pipeline: {str(e)}")
        raise

# Initialize the pipeline
try:
    pipeline = create_default_pipeline()
except Exception as e:
    logger.critical(f"Failed to initialize pipeline: {str(e)}")
    raise

# Create an event emitter function for a specific client
def create_emitter(client_id: str) -> Callable[[Dict[str, Any]], None]:
    async def emit_event(event: Dict[str, Any]):
        try:
            await manager.send_update(client_id, event)
        except Exception as e:
            logger.error(f"Error in emit_event for client {client_id}: {str(e)}")
    
    # Return a function that ignores the awaitable (for compatibility with synchronous code)
    def emitter(event: Dict[str, Any]):
        try:
            asyncio.create_task(emit_event(event))
        except Exception as e:
            logger.error(f"Error creating emit_event task for client {client_id}: {str(e)}")
        
    return emitter

# Modified search method with progress reporting
async def search_with_progress(query: str, client_id: str) -> Dict[str, Any]:
    """Run the pipeline with progress reporting"""
    logger.info(f"Starting search with progress for query: '{query}', client: {client_id}")
    emitter = create_emitter(client_id)
    
    # Initial status
    emitter({
        "type": "status",
        "data": {
            "status": "in_progress",
            "description": f"Processing query: {query}",
            "done": False,
            "action": "search_start",
            "urls": [],
        }
    })
    
    try:
        # Search
        emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": f"Searching for results",
                "done": False,
                "action": "search",
                "urls": [],
            }
        })
        search_results = await pipeline.search(query)
        logger.info(f"Search completed for query: '{query}', found {len(search_results)} results")
        
        # Extract URLs
        urls = pipeline._extract_urls(search_results)[:pipeline.max_sources]
        logger.info(f"Extracted {len(urls)} URLs for processing")
        
        # Scrape
        emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": f"Scraping content from {len(urls)} sources",
                "done": False,
                "action": "scrape",
                "urls": urls,
            }
        })
        scraped_content = await pipeline.scrape_content(search_results)
        logger.info(f"Content scraped from {len(scraped_content)} sources")
        
        # Process
        emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": "Processing and ranking content",
                "done": False,
                "action": "process",
                "urls": urls,
            }
        })
        processed_content = await pipeline.process_content(scraped_content, query)
        logger.info(f"Content processed, {len(processed_content)} items after processing")
        
        # Format as API response
        results = []
        for item in processed_content:
            result = {
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "snippet": item.get("snippet", ""),
            }
            if "score" in item:
                result["score"] = item["score"]
            results.append(result)
        
        # Final status
        emitter({
            "type": "status",
            "data": {
                "status": "complete",
                "description": f"Search completed with {len(results)} results",
                "done": True,
                "action": "search_complete",
                "urls": urls,
            }
        })
        
        logger.info(f"Search with progress completed successfully for client: {client_id}")
        return {
            "results": results,
            "query": query
        }
    except Exception as e:
        logger.error(f"Error in search_with_progress for query '{query}', client {client_id}: {str(e)}")
        # Notify client about the error
        emitter({
            "type": "error",
            "data": {
                "status": "error",
                "description": f"Search failed: {str(e)}",
                "done": True,
                "action": "search_error",
            }
        })
        raise

# Regular HTTP endpoint for search
@app.post("/search", response_model=SearchResponse)
async def search_endpoint(query: SearchQuery, request: Request):
    client_id = str(uuid.uuid4())
    logger.info(f"Search request received: '{query.query}', assigned client ID: {client_id}")
    try:
        response = await search_with_progress(query.query, client_id)
        return response
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time progress updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        await manager.connect(websocket, client_id)
        try:
            while True:
                # Just keep the connection alive and wait for events
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for client: {client_id}")
            manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {str(e)}")
            manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection for client {client_id}: {str(e)}")

# Server-sent events (SSE) endpoint for progress updates
@app.get("/events/{client_id}")
async def sse_endpoint(client_id: str):
    logger.info(f"SSE connection requested for client: {client_id}")
    try:
        queue = await manager.register_sse(client_id)
        
        async def event_generator():
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield json.dumps(event)
            except Exception as e:
                logger.error(f"Error in SSE event generator for client {client_id}: {str(e)}")
            finally:
                logger.info(f"SSE event generator closing for client: {client_id}")
                manager.unregister_sse(client_id)
        
        return EventSourceResponse(event_generator())
    except Exception as e:
        logger.error(f"Error setting up SSE for client {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

# Run the server when executed directly
if __name__ == "__main__":
    logger.info("Starting RAGSearch API server")
    uvicorn.run(app, host="0.0.0.0", port=8222) 