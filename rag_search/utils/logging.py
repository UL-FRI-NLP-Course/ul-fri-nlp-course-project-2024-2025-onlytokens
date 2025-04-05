from typing import Any, Dict, List, Optional
from termcolor import colored

# Map of component names to colors for visual distinction
COMPONENT_COLORS = {
    "QueryEnhancer": "cyan",
    "SearXNGProvider": "yellow",
    "WebScraper": "magenta",
    "Chunker": "blue",
    "SentenceEmbedder": "green",
    "Reranker": "light_magenta",
    "CrossEncoder": "light_magenta",
    "ContextBuilder": "yellow",
    "OpenAI": "green",
    "LLM": "green",
    "RAGSearchPipeline": "light_magenta"
}

def get_component_color(component: str) -> str:
    """Get color for a component, with fallback to white"""
    return COMPONENT_COLORS.get(component, "white")

def log_operation_start(title: str, component: str):
    """Log the start of an operation with a colored banner."""
    color = get_component_color(component)
    print(colored(f"\n▶ {title} [{component}]", color, attrs=["bold"]))

def log_operation_end(title: str, component: str):
    """Log the end of an operation with a colored banner."""
    color = get_component_color(component)
    print(colored(f"✓ {title} DONE [{component}]", color, attrs=["bold"]))

def log_info(message: str, component: str):
    """Log general information."""
    color = get_component_color(component)
    print(colored(f"  [{component}] {message}", color))

def log_data(key: str, value: Any, component: str):
    """Log a data key-value pair."""
    color = get_component_color(component)
    if isinstance(value, list):
        value_str = str(value)
    elif isinstance(value, dict):
        value_str = str(value)
    else:
        value_str = str(value)
    
    print(colored(f"  [{component}] {key}: {value_str}", color))

def log_input(data: Any, component: str):
    """Log input data with a specific format."""
    color = get_component_color(component)
    if isinstance(data, str):
        data_str = data
    else:
        data_str = str(data)
    print(colored(f"  [{component}] INPUT: {data_str}", color, attrs=["bold"]))

def log_output(data: Any, component: str):
    """Log output data with a specific format."""
    color = get_component_color(component)
    if isinstance(data, str):
        data_str = data
    elif isinstance(data, list):
        data_str = str(data)
    else:
        data_str = str(data)
    print(colored(f"  [{component}] OUTPUT: {data_str}", color, attrs=["bold"]))

def log_error(message: str, component: str, error: Optional[Exception] = None):
    """Log error information."""
    print(colored(f"\n❌ [{component}] ERROR: {message}", "red", attrs=["bold"]))
    if error:
        print(colored(f"  Details: {str(error)}", "red"))

def log_success(message: str, component: str):
    """Log success information."""
    color = get_component_color(component)
    print(colored(f"  [{component}] ✓ {message}", color))

def log_warning(message: str, component: str):
    """Log warning information."""
    print(colored(f"  [{component}] ⚠️ {message}", "yellow"))

def log_search_results(results: Dict[str, Any], component: str):
    """Log search results in a structured way."""
    color = get_component_color(component)
    organic_count = len(results.get("organic", []))
    
    print(colored(f"  [{component}] FOUND: {organic_count} results", color, attrs=["bold"]))
    
    # Display all results without truncation
    for i, result in enumerate(results.get("organic", [])):
        print(colored(f"  [{component}] #{i+1}: {result.get('title', 'No title')}", color))
        print(colored(f"        URL: {result.get('link', 'No link')}", "white"))
        print(colored(f"        Snippet: {result.get('snippet', 'No snippet')}", "white"))

def log_embedding_operation(texts_count: int, component: str, dim: Optional[int] = None):
    """Log embedding operation."""
    color = get_component_color(component)
    if dim:
        print(colored(f"  [{component}] EMBEDDING: {texts_count} texts with dimension {dim}", color, attrs=["bold"]))
    else:
        print(colored(f"  [{component}] EMBEDDING: {texts_count} texts", color, attrs=["bold"]))

def log_chunks(chunks: List[Dict[str, Any]], component: str, max_display: int = None):
    """Log chunks in a structured way."""
    color = get_component_color(component)
    print(colored(f"  [{component}] CHUNKS: {len(chunks)} total", color, attrs=["bold"]))
    
    # Display all chunks without limiting to max_display
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        url = chunk.get("url", chunk.get("metadata", {}).get("url", "Unknown source"))
        similarity = chunk.get("similarity", None)
        
        print(colored(f"  [{component}] Chunk #{i+1}: {content}", color))
        print(colored(f"        Source: {url}", "white"))
        if similarity is not None:
            print(colored(f"        Relevance: {similarity:.4f}", "white")) 