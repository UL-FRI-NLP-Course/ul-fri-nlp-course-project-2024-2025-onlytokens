# Natural language processing course: `OnlyTokens`

## About 

This project aims to develop an advanced conversational agent that leverages Retrieval-Augmented Generation
(RAG) to enhance response quality and factual accuracy. Unlike traditional chatbots relying solely on pre-trained
knowledge, our system will incorporate real-time web scraping capabilities to retrieve up-to-date information from
the internet. By dynamically fetching and integrating external information during conversation, the agent can
provide more accurate, comprehensive, and current responses. 

### Setup

install uv 

```
curl -fsSL https://get.uv.dev | sh
```

install dependencies

```
uv sync
```

### Start SearXNG

1. Navigate to the SearXNG directory:
```
cd searxng-docker
```

1. Edit the environment file:
```
touch .env && echo "SEARXNG_HOSTNAME=localhost" >> .env
```

1. Start SearXNG using Docker Compose:
```
docker-compose up -d
```

1. SearXNG will be available at `http://localhost:5555`


### Run the pipeline with OpenAI instead of local model

```
uv run main.py --config config.yaml --use-openai
```
if using OpenAI, you need to set the `OPENAI_API_KEY` environment variable:
```
export OPENAI_API_KEY=<your-openai-api-key>
```

### Run the pipeline with local model

```
uv run main.py --config config.yaml
```


## Diagram

![Diagram](./docs/diagram.png)
