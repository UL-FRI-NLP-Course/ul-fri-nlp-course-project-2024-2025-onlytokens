from firecrawl import FirecrawlApp

app = FirecrawlApp(api_key="fc-YOUR_API_KEY",api_url="http://localhost:3002")

# # Scrape a website:
# scrape_result = app.scrape_url('https://www.facebook.com/luka.dragar.1', params={'formats': ['markdown', 'html']})
# print(scrape_result)

from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field

# Initialize the FirecrawlApp with your API key



# data = app.extract([
#   'https://docs.firecrawl.dev/*', 
#   'https://firecrawl.dev/', 
#   'https://www.ycombinator.com/companies/'
# ], {
#     'prompt': 'Extract the company mission, whether it supports SSO, whether it is open source, and whether it is in Y Combinator from the page.',
# })
# print(data)

url = "https://www.facebook.com/luka.dragar.1"
result = app.scrape_url(url)
print(result)