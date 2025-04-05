from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """A modular text chunking class that splits text into smaller, overlapping segments.
    
    This class provides a flexible way to break down large texts into smaller chunks
    while maintaining context through configurable overlap. It uses RecursiveCharacterTextSplitter
    from langchain under the hood.
    
    Attributes:
        chunk_size (int): The target size for each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        separators (List[str]): List of separators to use for splitting, in order of preference.
        length_function (callable): Function to measure text length (default: len).
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: callable = len,
        verbose: bool = False
    ):
        """Initialize the Chunker with specified parameters.
        
        Args:
            chunk_size (int, optional): Target size for each chunk. Defaults to 250.
            chunk_overlap (int, optional): Number of characters to overlap. Defaults to 50.
            separators (List[str], optional): Custom separators for splitting.
                Defaults to ["\n\n", "\n"].
            length_function (callable, optional): Function to measure text length.
                Defaults to len.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n"]
        self.length_function = length_function
        self.verbose = verbose
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split a single text into chunks.
        
        Args:
            text (str): The input text to be split into chunks.
            
        Returns:
            List[str]: A list of text chunks.
        """
        return self.splitter.split_text(text)
    
    def split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into chunks.
        
        Args:
            texts (List[str]): A list of input texts to be split into chunks.
            
        Returns:
            List[List[str]]: A list of lists, where each inner list contains
                the chunks for one input text.
        """
        return [self.split_text(text) for text in texts]
    
    def split_scraped_content(self, scraped_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split scraped content into chunks while preserving metadata.
        
        Args:
            scraped_content (List[Dict[str, Any]]): A list of dictionaries containing
                scraped content with 'url' and 'content' fields.
                
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing chunked content
                with preserved metadata about the source.
        """
        chunks = []
        
        for idx, content_item in enumerate(scraped_content):
            # Get content from the 'no_extraction' strategy if available, otherwise use the first available
            if isinstance(content_item, dict) and 'no_extraction' in content_item:
                source_content = content_item['no_extraction'].content if content_item['no_extraction'].success else None
                url = content_item['no_extraction'].url
                metadata = content_item['no_extraction'].metadata or {}
            elif isinstance(content_item, dict) and content_item:
                # Get the first successful strategy
                for strategy_name, strategy_result in content_item.items():
                    if strategy_result.success and strategy_result.content:
                        source_content = strategy_result.content
                        url = strategy_result.url
                        metadata = strategy_result.metadata or {}
                        break
                else:
                    # No successful strategy found
                    continue
            else:
                # Fallback for simple dictionaries with direct content
                source_content = content_item.get('content')
                url = content_item.get('url')
                metadata = content_item.get('metadata', {})
            
            if not source_content:
                continue
                
            # Split the content into chunks
            text_chunks = self.split_text(source_content)
            
            if self.verbose:
                print(f"[DEBUG] Split source {idx} into {len(text_chunks)} chunks")
            
            # Create chunk objects with source metadata
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk_obj = {
                    'content': chunk_text,
                    'url': url,
                    'chunk_index': chunk_idx,
                    'source_index': idx,
                    'total_chunks': len(text_chunks)
                }
                
                # Preserve any additional metadata
                if metadata:
                    chunk_obj['metadata'] = metadata
                
                chunks.append(chunk_obj)
        
        return chunks
        
        
