import re
import wikipediaapi

def get_wikipedia_content(url: str) -> str | None:
    """
    Extract content from a Wikipedia URL.
    
    Args:
        url: Wikipedia URL to scrape
        
    Returns:
        str: Page content if found, None otherwise
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="FRI-NLP-Project/1.0 (University of Ljubljana Faculty of Computer and Information Science NLP Course; https://github.com/ul-fri-nlp-course-project-2024-2025-onlytokens)",
        language='en'
    )
    
    # Extract the page title from URL (everything after /wiki/)
    try:
        title = url.split('/wiki/')[-1]
        page = wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception:
        return None

# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"
IFRAME_PATTERN = r"<[ ]*iframe.*?\/[ ]*iframe[ ]*>"
NOSCRIPT_PATTERN = r"<[ ]*noscript.*?\/[ ]*noscript[ ]*>"
HEADER_PATTERN = r"<[ ]*header.*?\/[ ]*header[ ]*>"
FOOTER_PATTERN = r"<[ ]*footer.*?\/[ ]*footer[ ]*>"
NAV_PATTERN = r"<[ ]*nav.*?\/[ ]*nav[ ]*>"
FORM_PATTERN = r"<[ ]*form.*?\/[ ]*form[ ]*>"


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    """Clean HTML content by removing various elements."""
    patterns = [
        SCRIPT_PATTERN,
        STYLE_PATTERN,
        META_PATTERN,
        COMMENT_PATTERN,
        LINK_PATTERN,
        IFRAME_PATTERN,
        NOSCRIPT_PATTERN,
        HEADER_PATTERN,
        FOOTER_PATTERN,
        NAV_PATTERN,
        FORM_PATTERN
    ]
    
    for pattern in patterns:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
        
    # Remove empty lines and excessive whitespace
    html = re.sub(r'\n\s*\n', '\n', html)
    html = re.sub(r'\s+', ' ', html)
    
    return html.strip()

JSON_SCHEMA = """
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string"
    },
    "author": {
      "type": "string"
    },
    "date": {
      "type": "string"
    },
    "content": {
      "type": "string"
    }
  },
  "required": ["title", "author", "date", "content"]
}
"""