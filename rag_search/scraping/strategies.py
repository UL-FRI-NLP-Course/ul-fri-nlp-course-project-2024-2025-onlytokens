from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re

class ExtractionStrategy(ABC):
    """Abstract base class for content extraction strategies."""
    
    @abstractmethod
    def extract(self, html_content: str, url: str) -> str:
        """
        Extract content from HTML.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Extracted content as string
        """
        pass

class NoExtractionStrategy(ExtractionStrategy):
    """Strategy that performs minimal cleaning without special extraction."""
    
    def extract(self, html_content: str, url: str) -> str:
        """
        Extract content with minimal processing.
        
        Simply converts HTML to text with basic cleaning.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "noscript", "svg", "path"]):
            script.decompose()
            
        # Get text and normalize whitespace
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class CSSExtractionStrategy(ExtractionStrategy):
    """Strategy that extracts content using CSS selectors."""
    
    # Common content selectors for different website types
    DEFAULT_SELECTORS = [
        # News and article sites
        "article", "main", ".article-content", ".post-content", ".entry-content",
        # Generic content areas
        "#content", ".content", "#main", ".main", ".body", "#body",
        # Blog-specific
        ".blog-post", ".blog-entry", ".post"
    ]
    
    def __init__(self, custom_selectors: Optional[List[str]] = None):
        """
        Initialize with optional custom selectors.
        
        Args:
            custom_selectors: List of CSS selectors to try
        """
        self.selectors = custom_selectors or self.DEFAULT_SELECTORS
    
    def extract(self, html_content: str, url: str) -> str:
        """
        Extract content using CSS selectors.
        
        Tries multiple selectors and uses the one that yields the most content.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try each selector and find the one with the most content
        best_content = ""
        for selector in self.selectors:
            elements = soup.select(selector)
            for element in elements:
                # Clean the element
                for script in element(["script", "style", "meta", "noscript", "svg", "path"]):
                    script.decompose()
                
                content = element.get_text(separator=' ')
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Keep the longest content
                if len(content) > len(best_content):
                    best_content = content
        
        # If no content was found with selectors, fall back to basic extraction
        if not best_content:
            return NoExtractionStrategy().extract(html_content, url)
            
        return best_content

class XPathExtractionStrategy(ExtractionStrategy):
    """Strategy that extracts content using XPath expressions."""
    
    # Common XPath expressions for content extraction
    DEFAULT_XPATHS = [
        "//article",
        "//main",
        "//*[contains(@class, 'content')]",
        "//*[contains(@class, 'article')]",
        "//*[contains(@class, 'post')]",
        "//div[@id='content']",
        "//div[@id='main']"
    ]
    
    def __init__(self, custom_xpaths: Optional[List[str]] = None):
        """
        Initialize with optional custom XPath expressions.
        
        Args:
            custom_xpaths: List of XPath expressions to try
        """
        self.xpaths = custom_xpaths or self.DEFAULT_XPATHS
        
    def extract(self, html_content: str, url: str) -> str:
        """
        Extract content using XPath expressions.
        
        Note: BeautifulSoup doesn't directly support XPath, so we'll simulate
        XPath-like behavior using CSS selectors where possible.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Helper function to convert simplified XPath to CSS selector
        def xpath_to_css(xpath: str) -> str:
            # Very basic conversion for common patterns
            if xpath.startswith("//"):
                xpath = xpath[2:]  # Remove leading //
            # Convert id and class expressions
            xpath = xpath.replace("[@id='", "#").replace("']", "")
            xpath = xpath.replace("[contains(@class, '", ".").replace("')]", "")
            return xpath
        
        # Try each XPath-like expression
        best_content = ""
        for xpath in self.xpaths:
            try:
                # Convert to CSS selector where possible
                css_selector = xpath_to_css(xpath)
                elements = soup.select(css_selector)
                
                for element in elements:
                    # Clean the element
                    for script in element(["script", "style", "meta", "noscript", "svg", "path"]):
                        script.decompose()
                    
                    content = element.get_text(separator=' ')
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    # Keep the longest content
                    if len(content) > len(best_content):
                        best_content = content
            except Exception:
                # Skip errors in XPath conversion
                continue
        
        # If no content was found, fall back to basic extraction
        if not best_content:
            return NoExtractionStrategy().extract(html_content, url)
            
        return best_content
