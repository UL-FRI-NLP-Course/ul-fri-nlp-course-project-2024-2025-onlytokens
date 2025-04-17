"""
Contains the StrategyFactory class for creating various extraction strategies.
"""

import os
from typing import Any, Dict, List, Optional

from crawl4ai import ExtractionStrategy, LLMConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    NoExtractionStrategy,
    CosineStrategy,
)

class LukasExtractionStrategy(ExtractionStrategy):
    """
    A strategy that does not extract any meaningful content from the HTML. It simply returns the entire HTML as a single block.
    """

    def extract(self, url: str, html: str, *q, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract meaningful blocks or chunks from the given HTML.
        """
        return [{"index": 0, "content": html}]

    def run(self, url: str, sections: List[str], *q, **kwargs) -> List[Dict[str, Any]]:
        return [
            {"index": i, "tags": [], "content": section}
            for i, section in enumerate(sections)
        ]


class StrategyFactory:
    """Factory for creating extraction strategies"""



    @staticmethod
    def create_lukas_strategy() -> LukasExtractionStrategy:
        #NOTE we use this so we get the fit_markdown output which means filtered markdown this make sthe piepline use the applied filters
        return LukasExtractionStrategy(input_format="fit_markdown")
    





    @staticmethod
    def create_llm_strategy(
        input_format: str = "fit_markdown",
        instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        base_url: str = "https://localhost:8001/v1/",
    ) -> LLMExtractionStrategy:
        #TODO READ ON THIS in crawl4ai allpy chunking... ...
        return LLMExtractionStrategy(
            input_format=input_format,
            #TODO MAYBE CHANGE PARAMSM HERE LIKE REDUCE THE TEMPERATURE AND MAX TOKENS also check if crawl ai does that internally
            llm_config=LLMConfig(provider="openai/nemotron",base_url=base_url,api_token="sk-or-v1-1234567890"),
            instruction=instruction,
            verbose=True
        )

    @staticmethod
    def create_css_strategy() -> JsonCssExtractionStrategy:
        schema = {
            "baseSelector": ".product",
            "fields": [
                {"name": "title", "selector": "h1.product-title", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"},
                {"name": "description", "selector": ".description", "type": "text"},
            ],
        }
        return JsonCssExtractionStrategy(schema=schema)

    @staticmethod
    def create_xpath_strategy() -> JsonXPathExtractionStrategy:
        schema = {
            "baseSelector": "//div[@class='product']",
            "fields": [
                {"name": "title", "selector": ".//h1[@class='product-title']/text()", "type": "text"},
                {"name": "price", "selector": ".//span[@class='price']/text()", "type": "text"},
                {"name": "description", "selector": ".//div[@class='description']/text()", "type": "text"},
            ],
        }
        return JsonXPathExtractionStrategy(schema=schema)

    @staticmethod
    def create_no_extraction_strategy() -> NoExtractionStrategy:
        return NoExtractionStrategy()
    


    @staticmethod
    def create_cosine_strategy(
        semantic_filter: Optional[str] = None,
        word_count_threshold: int = 10,
        max_dist: float = 0.2,
        sim_threshold: float = 0.3,
        debug: bool = False
    ) -> CosineStrategy:
        return CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=word_count_threshold,
            max_dist=max_dist,
            sim_threshold=sim_threshold,
            verbose=debug
        ) 