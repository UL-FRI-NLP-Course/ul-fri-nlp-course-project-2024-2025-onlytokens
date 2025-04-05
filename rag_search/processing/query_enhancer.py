from typing import List, Optional
import re
from dataclasses import dataclass

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_success
)

@dataclass
class EnhancedQuery:
    original_query: str
    enhanced_query: str
    search_keywords: List[str]
    

class QueryEnhancer:
    """
    A class to enhance search queries by extracting important keywords,
    expanding them with synonyms, or reformulating for better search results.
    """
    
    def __init__(
        self,
        max_keywords: int = 5,
        min_keyword_length: int = 3,
        remove_stopwords: bool = True,
        verbose: bool = False
    ):
        """
        Initialize query enhancer.
        
        Args:
            max_keywords: Maximum number of keywords to extract
            min_keyword_length: Minimum length for a keyword
            remove_stopwords: Whether to remove stopwords
            verbose: Whether to enable verbose logging
        """
        self.max_keywords = max_keywords
        self.min_keyword_length = min_keyword_length
        self.remove_stopwords = remove_stopwords
        self.verbose = verbose
        
        # Common English stopwords
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "having", "do", "does", "did", "doing",
            "would", "should", "could", "ought", "i", "you", "he", "she", "it",
            "we", "they", "their", "his", "her", "its", "our", "your", "my"
        }
        
        if self.verbose:
            log_info("QueryEnhancer ready!", "QueryEnhancer")
    
    def enhance(self, query: str) -> EnhancedQuery:
        """
        Enhance the given query by extracting keywords, removing stopwords,
        and reformulating it for better search results.
        
        Args:
            query: The original user query
            
        Returns:
            EnhancedQuery object containing original query, enhanced query,
            and extracted keywords
        """
        if self.verbose:
            log_operation_start("ENHANCING QUERY", "QueryEnhancer")
            log_input(query, "QueryEnhancer")
        
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_query)
        if self.verbose:
            log_info(f"Extracted keywords: {', '.join(keywords)}", "QueryEnhancer")
        
        # Create an enhanced query
        enhanced_query = self._reformulate_query(cleaned_query, keywords)
        
        result = EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            search_keywords=keywords
        )
        
        if self.verbose:
            log_success("Query enhanced", "QueryEnhancer")
            log_output(enhanced_query, "QueryEnhancer")
            log_operation_end("ENHANCING QUERY", "QueryEnhancer")
            
        return result
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove any special characters except spaces and alphanumerics
        query = re.sub(r'[^\w\s]', '', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Split into words
        words = query.split()
        
        # Filter words
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        
        # Filter by length
        words = [w for w in words if len(w) >= self.min_keyword_length]
        
        # Limit to max keywords
        return words[:self.max_keywords]
    
    def _reformulate_query(self, query: str, keywords: List[str]) -> str:
        """
        Reformulate the query for better search results.
        
        For now, this is a simple implementation that adds quotes around
        the original query and appends a few extracted keywords.
        This can be expanded with more sophisticated techniques.
        """
        # Start with the original query
        enhanced = query
        
        # If we have keywords, add them
        if keywords:
            # In this simple version, we'll just add the most important keywords
            # with explicit inclusion markers
            keyword_str = " ".join([f"+{k}" for k in keywords[:3]])
            enhanced = f"{enhanced} {keyword_str}"
            
        return enhanced
    
if __name__ == "__main__":
    query_enhancer = QueryEnhancer(verbose=True)
    enhanced_query = query_enhancer.enhance("Im trying to find a good restaurant in the center of town in Ljubljana where should I go on a nice date?")
    print(enhanced_query)
