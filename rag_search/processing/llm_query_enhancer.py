from typing import List
from dataclasses import dataclass
from pydantic import BaseModel, Field
import openai
from datetime import datetime
import os

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_success
)

class SearchQueriesResponse(BaseModel):
    """Pydantic model for the LLM response format"""
    queries: List[str] 

@dataclass
class EnhancedQueries:
    original_query: str
    enhanced_queries: List[str]

class LLMQueryEnhancer:
    """
    A class that uses LLM to generate multiple optimized search queries
    from a single user query.
    """
    
    def __init__(
        self,
        openai_client: openai.OpenAI,
        model: str = "gpt-3.5-turbo",
        max_queries: int = 3,
        verbose: bool = False
    ):
        """
        Initialize LLM query enhancer.
        
        Args:
            openai_client: OpenAI client instance
            model: Model to use for query enhancement
            max_queries: Maximum number of queries to generate
            verbose: Whether to enable verbose logging
        """
        self.client = openai_client
        self.model = model
        self.max_queries = max_queries
        self.verbose = verbose
        
        if self.verbose:
            log_info("LLMQueryEnhancer ready!", "LLMQueryEnhancer")

    def _get_system_prompt(self) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"""You are an expert search query generator. Your task is to analyze user queries and generate multiple optimized search queries that will help retrieve comprehensive and relevant information.

Current date: {current_date}

Guidelines for generating queries:
1. Generate diverse queries that cover different aspects of the user's information need
2. Use advanced search operators when beneficial (e.g., site:, filetype:, etc.)
3. Include temporal aspects if relevant (e.g., 2025, latest, current)
4. Consider both broad and specific queries to ensure good coverage
5. Maintain relevance to the original query intent
6. Use quotes for exact phrases when appropriate
7. Include relevant technical terms and synonyms
8. Generate queries in the same language as the input query (e.g., if user asks in Slovenian, generate Slovenian queries)
9. Preserve any language-specific characters and accents from the input language

You must respond with valid JSON matching the specified schema."""

    def _get_user_prompt(self, query: str) -> str:
        return f"""Generate {self.max_queries} optimized search queries for the following user query:

User Query: {query}

Generate queries that will help find the most relevant, comprehensive, and up-to-date information to answer this query.
Focus on creating diverse queries that cover different aspects while maintaining relevance to the original intent."""

    async def enhance(self, query: str) -> EnhancedQueries:
        """
        Enhance the given query by generating multiple optimized search queries using LLM.
        
        Args:
            query: The original user query
            
        Returns:
            EnhancedQueries object containing original query and list of enhanced queries
        """
        if self.verbose:
            log_operation_start("ENHANCING QUERY WITH LLM", "LLMQueryEnhancer")
            log_input(query, "LLMQueryEnhancer")
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._get_user_prompt(query)}
                ],
                response_format=SearchQueriesResponse
            )
                        
            # Parse the response
            content = response.choices[0].message.content
            queries_response = SearchQueriesResponse.model_validate_json(content)
            
            result = EnhancedQueries(
                original_query=query,
                enhanced_queries=queries_response.queries
            )
            
            if self.verbose:
                log_success("Queries enhanced with LLM", "LLMQueryEnhancer")
                log_output(result.enhanced_queries, "LLMQueryEnhancer")
                log_operation_end("ENHANCING QUERY WITH LLM", "LLMQueryEnhancer")
                
            return result
            
        except Exception as e:
            if self.verbose:
                log_info(f"Error enhancing query: {str(e)}", "LLMQueryEnhancer")
            # Fallback to original query if enhancement fails
            return EnhancedQueries(
                original_query=query,
                enhanced_queries=[query]
            )

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Example usage
    load_dotenv()
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    enhancer = LLMQueryEnhancer(client, verbose=True)
    query = "Kdo je Luka Dragar? in zakaj je znan?"
    
    # Run async function
    enhanced = asyncio.run(enhancer.enhance(query))
    print(enhanced) 