from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_success, log_warning
)

class ContextBuilder(ABC):
    """Base class for context builders."""
    
    @abstractmethod
    def build(
        self,
        processed_content: List[Dict[str, Any]],
        search_results: Dict[str, Any]
    ) -> str:
        """
        Build context from processed content and search results.
        
        Args:
            processed_content: List of processed content chunks
            search_results: Original search results
            
        Returns:
            Formatted context string
        """
        pass

class SimpleContextBuilder(ContextBuilder):
    """Simple context builder that formats chunks with minimal processing."""
    
    def __init__(
        self,
        include_answer_box: bool = True,
        include_metadata: bool = True,
        chunk_separator: str = "\n\n",
        verbose: bool = False
    ):
        """
        Initialize simple context builder.
        
        Args:
            include_answer_box: Whether to include answer box in context
            include_metadata: Whether to include source metadata
            chunk_separator: Separator between content chunks
            verbose: Whether to enable verbose logging
        """
        self.include_answer_box = include_answer_box
        self.include_metadata = include_metadata
        self.chunk_separator = chunk_separator
        self.verbose = verbose
        
        if self.verbose:
            log_info("Context builder ready!", "ContextBuilder")
    
    def build(
        self,
        processed_content: List[Dict[str, Any]],
        search_results: Dict[str, Any]
    ) -> str:
        """
        Build context from processed content and search results.
        
        Args:
            processed_content: List of processed content chunks
            search_results: Original search results
            
        Returns:
            Formatted context string
        """
        if self.verbose:
            log_operation_start("BUILDING CONTEXT", "ContextBuilder")
            log_input(f"{len(processed_content)} content chunks", "ContextBuilder")
            
        context_parts = []
        
        # Add answer box if available and requested
        if self.include_answer_box and search_results.get('answerBox'):
            if self.verbose:
                log_info("Processing answer box", "ContextBuilder")
                
            answer_box = search_results['answerBox']
            answer_box_content = []
            
            for key in ['title', 'answer', 'snippet']:
                if answer_box.get(key):
                    answer_box_content.append(answer_box[key])
            
            if answer_box_content:
                context_parts.append("ANSWER BOX:")
                context_parts.append("\n".join(answer_box_content))
                context_parts.append("")  # Empty line for separation
                
                if self.verbose:
                    log_success("Added answer box to context", "ContextBuilder")
        
        # Add processed content chunks
        if processed_content:
            if self.verbose:
                log_info(f"Adding {len(processed_content)} content chunks", "ContextBuilder")
                
            context_parts.append("RELEVANT CONTENT:")
            
            for chunk in processed_content:
                chunk_parts = []
                
                # Add content
                if 'content' in chunk:
                    chunk_parts.append(chunk['content'])
                
                # Add metadata if requested
                if self.include_metadata and 'metadata' in chunk:
                    metadata = chunk['metadata']
                    url = chunk.get('url', metadata.get('url', 'Unknown source'))
                    
                    # Format source info
                    source_info = f"Source: {url}"
                    
                    # Add similarity score if available
                    if 'similarity' in chunk:
                        score = chunk['similarity']
                        source_info += f" (Relevance: {score:.2f})"
                        
                    chunk_parts.append(source_info)
                
                # Add the formatted chunk
                if chunk_parts:
                    context_parts.append("\n".join(chunk_parts))
        
        # Combine all parts with separators
        result = self.chunk_separator.join(context_parts)
        
        if self.verbose:
            log_success("Context built successfully", "ContextBuilder")
            log_output(f"Context with {len(result)} characters", "ContextBuilder")
            log_operation_end("BUILDING CONTEXT", "ContextBuilder")
            
        return result

class StructuredContextBuilder(ContextBuilder):
    """Context builder with more structured formatting."""
    
    def __init__(
        self,
        max_answer_box_length: int = 1000,
        max_chunk_length: int = 1000,
        total_context_limit: int = 8000,
        include_related_searches: bool = True,
        verbose: bool = False
    ):
        """
        Initialize structured context builder.
        
        Args:
            max_answer_box_length: Maximum length for answer box section
            max_chunk_length: Maximum length for each content chunk
            total_context_limit: Maximum total context length
            include_related_searches: Whether to include related searches
            verbose: Whether to enable verbose logging
        """
        self.max_answer_box_length = max_answer_box_length
        self.max_chunk_length = max_chunk_length
        self.total_context_limit = total_context_limit
        self.include_related_searches = include_related_searches
        self.verbose = verbose
        
        if self.verbose:
            log_info("Structured context builder ready!", "ContextBuilder")
        
    def build(
        self,
        processed_content: List[Dict[str, Any]],
        search_results: Dict[str, Any]
    ) -> str:
        """
        Build context from processed content and search results with structure.
        
        Args:
            processed_content: List of processed content chunks
            search_results: Original search results
            
        Returns:
            Formatted context string with sections
        """
        if self.verbose:
            log_operation_start("BUILDING STRUCTURED CONTEXT", "ContextBuilder")
            log_input(f"{len(processed_content)} chunks, limit: {self.total_context_limit} chars", "ContextBuilder")
            
        sections = []
        current_length = 0
        
        # 1. Add answer box section
        answer_box_section = self._format_answer_box(search_results)
        if answer_box_section:
            sections.append(answer_box_section)
            current_length += len(answer_box_section)
            
            if self.verbose:
                log_success("Added answer box section", "ContextBuilder")
        
        # 2. Add content chunks section
        if processed_content and current_length < self.total_context_limit:
            if self.verbose:
                log_info(f"Adding content chunks with {self.total_context_limit - current_length} chars available", "ContextBuilder")
                
            content_section = self._format_content_chunks(
                processed_content,
                max_length=self.total_context_limit - current_length
            )
            if content_section:
                sections.append(content_section)
                current_length += len(content_section)
                
                if self.verbose:
                    log_success(f"Added content section with {len(content_section)} chars", "ContextBuilder")
        
        # 3. Add related searches if requested and space allows
        if (self.include_related_searches and 
            search_results.get('relatedSearches') and
            current_length < self.total_context_limit):
            
            if self.verbose:
                log_info(f"Adding related searches with {self.total_context_limit - current_length} chars available", "ContextBuilder")
            
            related_section = self._format_related_searches(
                search_results['relatedSearches'],
                max_length=self.total_context_limit - current_length
            )
            if related_section:
                sections.append(related_section)
                current_length += len(related_section)
                
                if self.verbose:
                    log_success(f"Added related searches section with {len(related_section)} chars", "ContextBuilder")
        
        # Combine all sections
        result = "\n\n".join(sections)
        
        if self.verbose:
            log_success("Structured context built successfully", "ContextBuilder")
            log_output(f"Final context: {len(result)} chars", "ContextBuilder")
            log_operation_end("BUILDING STRUCTURED CONTEXT", "ContextBuilder")
            
        return result
    
    def _format_answer_box(self, search_results: Dict[str, Any]) -> str:
        """Format answer box section."""
        if not search_results.get('answerBox'):
            return ""
            
        if self.verbose:
            log_info("Formatting answer box section", "ContextBuilder")
            
        answer_box = search_results['answerBox']
        parts = ["# Answer Box"]
        
        for key in ['title', 'answer', 'snippet']:
            if answer_box.get(key):
                parts.append(answer_box[key])
        
        result = "\n".join(parts)
        
        # No more truncation for answer box
        # if len(result) > self.max_answer_box_length:
        #     if self.verbose:
        #         log_warning(f"Answer box truncated to {self.max_answer_box_length} chars", "ContextBuilder")
        #     result = result[:self.max_answer_box_length - 3] + "..."
            
        return result
    
    def _format_content_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_length: int
    ) -> str:
        """Format content chunks section."""
        if self.verbose:
            log_info(f"Formatting {len(chunks)} content chunks", "ContextBuilder")
            
        parts = ["# Relevant Content"]
        current_length = len(parts[0])
        chunks_added = 0
        
        for chunk in chunks:
            # Format chunk with content and source
            chunk_parts = []
            
            # Add content
            content = chunk.get('content', '')
            if not content:
                continue
                
            # No more truncation for chunks
            # if len(content) > self.max_chunk_length:
            #     if self.verbose:
            #         log_info(f"Truncating chunk from {len(content)} to {self.max_chunk_length} chars", "ContextBuilder")
            #     content = content[:self.max_chunk_length - 3] + "..."
                
            chunk_parts.append(content)
            
            # Add source info
            url = chunk.get('url', chunk.get('metadata', {}).get('url', 'Unknown source'))
            source_info = f"Source: {url}"
            
            # Add similarity score if available
            if 'similarity' in chunk:
                score = chunk['similarity']
                source_info += f" (Relevance: {score:.2f})"
                
            chunk_parts.append(source_info)
            
            # Check if adding this chunk would exceed the limit
            chunk_text = "\n\n" + "\n".join(chunk_parts)
            # No more truncation based on maximum length
            # if current_length + len(chunk_text) > max_length:
            #     if self.verbose:
            #         log_warning(f"Reached length limit after {chunks_added} chunks", "ContextBuilder")
            #     break
                
            parts.append("\n".join(chunk_parts))
            current_length += len(chunk_text)
            chunks_added += 1
        
        if self.verbose:
            log_success(f"Added {chunks_added} of {len(chunks)} chunks", "ContextBuilder")
            
        return "\n\n".join(parts)
    
    def _format_related_searches(
        self,
        related_searches: List[str],
        max_length: int
    ) -> str:
        """Format related searches section."""
        if not related_searches:
            return ""
            
        if self.verbose:
            log_info(f"Formatting {len(related_searches)} related searches", "ContextBuilder")
            
        header = "# Related Searches"
        formatted_searches = "\n- " + "\n- ".join(related_searches)
        
        result = header + formatted_searches
        
        # No more truncation for related searches
        # if len(result) > max_length:
        #     if self.verbose:
        #         log_warning(f"Related searches exceed maximum length, truncating", "ContextBuilder")
        #         
        #     # Try to include as many complete related searches as possible
        #     parts = [header]
        #     current_length = len(header)
        #     searches_added = 0
        #     
        #     for search in related_searches:
        #         search_text = "\n- " + search
        #         if current_length + len(search_text) > max_length:
        #             if self.verbose:
        #                 log_info(f"Added {searches_added} of {len(related_searches)} related searches", "ContextBuilder")
        #             break
        #             
        #         parts.append(search_text)
        #         current_length += len(search_text)
        #         searches_added += 1
        #         
        #     result = "".join(parts)
            
        return result

class LukaContextBuilder(ContextBuilder):
    """Context builder that formats content with prompting instructions."""
    
    def __init__(
        self,
        include_scores: bool = True,
        verbose: bool = False
    ):
        """
        Initialize context builder.
        
        Args:
            include_scores: Whether to include relevance scores
            verbose: Whether to enable verbose logging
        """
        self.include_scores = include_scores
        self.verbose = verbose
        
        if self.verbose:
            log_info("Context builder ready!", "LukaContextBuilder")

    def _get_prompt_template(self) -> str:
        """Get the prompt template for the LLM."""
        return '''### Query:
{query}

### Context:
<context>
{context}
</context>

### Instructions:
You are a helpful AI assistant. Answer the query naturally and conversationally using the provided context.
- Use information from the context to support your answer
- Cite sources using [1], [2] etc. when referencing specific information
- Be concise and direct
- If you're not sure about something, say so
- If the context doesn't help answer the query, use your own knowledge but mention this
- Respond in the same language as the query
- Focus on being helpful rather than explaining your citations

Example of good response style:
"John Smith is a software engineer at Google [1] who specializes in machine learning. He previously worked at Microsoft [2] and has published several papers on AI safety."

### Response:'''

    def build(
        self,
        processed_content: List[Dict[str, Any]],
        search_results: Dict[str, Any]
    ) -> str:
        """
        Build context from processed content and search results using Luka formatting.
        
        Args:
            processed_content: List of processed content chunks
            search_results: Original search results (not used)
            
        Returns:
            Formatted context string with XML tags and prompting instructions
        """
        if self.verbose:
            log_operation_start("BUILDING CONTEXT", "LukaContextBuilder")
            log_input(f"{len(processed_content)} content chunks", "LukaContextBuilder")

        # Start building the context sections
        context_parts = []
        
        # Process content chunks
        if processed_content:
            if self.verbose:
                log_info(f"Adding {len(processed_content)} content chunks", "LukaContextBuilder")
            
            for i, chunk in enumerate(processed_content, 1):
                if 'content' not in chunk or 'url' not in chunk:
                    continue
                    
                # Build source tag with attributes
                source_attrs = [
                    f'url="{chunk["url"]}"',
                    f'id="{i}"'
                ]
                
                if self.include_scores and 'similarity' in chunk:
                    score = chunk['similarity']
                    source_attrs.append(f'relevance="{score:.2f}"')
                
                # Format the chunk with XML tags
                chunk_text = [
                    f"<source {' '.join(source_attrs)}>",
                    chunk['content'],
                    "</source>"
                ]
                
                context_parts.append("\n".join(chunk_text))

        # Combine all parts
        context = "\n\n".join(context_parts)
        
        if self.verbose:
            log_success("prompt built successfully", "LukaContextBuilder")
            log_output(f"Context with {len(context)} characters", "LukaContextBuilder")
            log_operation_end("BUILDING CONTEXT", "LukaContextBuilder")

        # Get the prompt template and format with context
        prompt_template = self._get_prompt_template()
        return prompt_template.format(
            context=context,
            query="{query}"  # This will be replaced by the actual query later
        )
