# RAG Pipeline Evaluation System

This directory contains the evaluation framework for the RAG (Retrieval-Augmented Generation) search pipeline. The evaluation system allows you to systematically test the pipeline's performance on a set of predefined questions and analyze the results.

## Overview

The evaluation system consists of:
- **`eval.py`**: Main evaluation script containing the `PipelineEvaluator` class
- **`analysis.py`**: Results analysis and visualization tools
- **`autograde_df.py`**: Automated grading utilities
- **Test sets**: CSV files containing questions, expected answers, and metadata
- **Results storage**: Organized output with logs, summaries, and detailed results

## Quick Start

### Basic Evaluation

```bash
# Run evaluation with default settings
cd eval
python eval.py

# Run with custom config and test set
python eval.py --config ../config.yaml --test-set ./simple_qa_test_set.csv
```

### Using OpenAI API

```bash
# Use OpenAI API instead of local endpoints
python eval.py --use-openai
```

### Resume Previous Run

```bash
# Continue a previous evaluation run
python eval.py --restore <run_id>
```

## Test Set Format

Test cases should be provided as a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `question` | The question to ask the pipeline | "What is the capital of France?" |
| `true_answer` | Expected/reference answer | "Paris" |
| `metadata` | Additional context as JSON string | "{'category': 'geography', 'difficulty': 'easy'}" |

### Example CSV:
```csv
question,true_answer,metadata
"What is the capital of France?","Paris","{'category': 'geography', 'difficulty': 'easy'}"
"Who wrote Romeo and Juliet?","William Shakespeare","{'category': 'literature', 'difficulty': 'medium'}"
```

## Configuration

The evaluator uses the same configuration system as the main pipeline. Key configuration sections:

### Pipeline Components
- **Query Enhancer**: Enhances input queries
- **Search Provider**: SearXNG-based web search
- **Web Scraper**: Content extraction from URLs
- **Chunker**: Text chunking for processing
- **Embedder**: Text embedding generation
- **Retriever**: Similarity-based retrieval
- **Reranker**: JinaAI-based reranking
- **Context Builder**: Final context assembly
- **LLM Provider**: Language model for generation

### Evaluation-Specific Settings
```yaml
pipeline:
  max_sources: 5  # Reduced for evaluation efficiency
  debug: false

# All component verbose flags are automatically set to false during evaluation
```

## Output Structure

Each evaluation run creates a timestamped directory with the following structure:

```
results/
└── eval_run_<uuid>/
    ├── run_metadata.json      # Run configuration and metadata
    ├── config.yaml           # Pipeline configuration used
    └── results.jsonl         # Detailed results for each test case

# Separate pipeline logs directory
pipeline_logs_<uuid>/
├── search_results/
├── scraped_content/
└── logs.jsonl
```

### Results Format

Each line in `results.jsonl` contains:

```json
{
  "case_id": "unique-case-identifier",
  "question": "Input question",
  "true_answer": "Expected answer",
  "pipeline_answer": "Generated answer",
  "enhanced_queries": ["enhanced", "query", "variants"],
  "metadata": {"category": "geography"},
  "retrieved_urls": ["url1", "url2"],
  "search_results": {"queries": {...}, "results": [...]},
  "success": true,
  "error": null
}
```

### Summary Statistics

The `summary.json` file includes:

```json
{
  "run_id": "evaluation-run-uuid",
  "timestamp": "20240101_120000",
  "total_cases": 50,
  "successful_cases": 48,
  "success_rate": 0.96,
  "average_urls_per_query": 8.5,
  "unique_urls_total": 234,
  "config_path": "./config.yaml",
  "test_set_path": "./test_set.csv"
}
```

## Results Analysis

The `analysis.py` script provides comprehensive visualization and analysis tools:

### Running Analysis

```bash
# Analyze results from a specific run
python analysis.py path/to/results.jsonl

# The script will generate interactive visualizations for:
# - Performance by topic (sunburst charts)
# - Performance by answer type
# - Grade distributions
# - Success rate analysis
```

### Generated Visualizations

The analysis creates several interactive HTML charts:
- **Topic Distribution**: Performance breakdown by question topics
- **Answer Type Distribution**: Performance by answer categories
- **Grade Distribution**: Overall performance metrics
- **Success Rate Analysis**: Detailed accuracy statistics

Charts are saved as both interactive HTML and high-resolution PDF files in the `report/fig/` directory.

## Advanced Usage

### Command Line Options

```bash
python eval.py [OPTIONS]

Options:
  --config PATH       Path to pipeline configuration file (default: ./config.yaml)
  --test-set PATH     Path to test set CSV file (default: ./simple_qa_test_set.csv)
  --results-dir PATH  Directory to store results (default: ./results)
  --use-openai        Use OpenAI API instead of local endpoints
  --restore RUN_ID    Continue evaluation from previous run
```

### Debugging

The evaluation script includes debugpy support for debugging:

```python
# The script waits for debugger attachment on port 5678
# Connect your IDE debugger to localhost:5678
```

To disable debugging for faster execution, comment out the debugpy lines in the script.

### Resuming Evaluations

If an evaluation is interrupted, you can resume it using the run ID:

1. Find the run ID from the results directory name
2. Use `--restore <run_id>` to continue from where it left off
3. Already processed questions will be skipped

## Pipeline Components

### Initialization Process

The evaluator initializes the full RAG pipeline with:

1. **LLM Query Enhancer**: Generates query variations
2. **SearXNG Search Provider**: Web search functionality
3. **Quality Improver**: Content quality assessment
4. **Web Scraper**: Multi-strategy content extraction
5. **Chunker**: Text segmentation
6. **OpenAI Embedder**: Vector embeddings
7. **Cosine Retriever**: Similarity search
8. **JinaAI Reranker**: Result reranking
9. **Context Builder**: Final context assembly
10. **OpenAI LLM Provider**: Answer generation

### State Management

Between test cases, the evaluator:
- Clears conversation history
- Resets cached content
- Clears component state
- Preserves logs for analysis

## Best Practices

### Test Set Design
1. **Diverse Questions**: Include various domains and difficulty levels
2. **Clear Answers**: Provide unambiguous expected answers
3. **Rich Metadata**: Include categorization for analysis
4. **Balanced Coverage**: Test different aspects of the pipeline

### Configuration
1. **Disable Verbose Logging**: Automatic during evaluation
2. **Appropriate Timeouts**: Set reasonable limits for web scraping
3. **Resource Limits**: Configure appropriate batch sizes
4. **Reduced Max Sources**: Use fewer sources for faster evaluation

### Analysis
1. **Success Rate**: Monitor overall pipeline reliability
2. **URL Coverage**: Analyze source diversity
3. **Error Patterns**: Identify common failure modes
4. **Performance Metrics**: Track response times and resource usage
5. **Topic Analysis**: Use visualization tools to identify weak areas

## Files in this Directory

- **`eval.py`**: Main evaluation script
- **`analysis.py`**: Results analysis and visualization
- **`autograde_df.py`**: Automated grading utilities
- **`config.yaml`**: Evaluation-specific configuration
- **`simple_qa_test_set.csv`**: Example test dataset
- **`results/`**: Directory containing evaluation results

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all pipeline components are available
2. **API Rate Limits**: Consider delays between requests
3. **Network Issues**: Check connectivity to search and LLM services
4. **Configuration Errors**: Validate YAML syntax and required fields
5. **Large Test Sets**: Use the restore functionality for interrupted runs

### Error Handling

The evaluator handles failures gracefully:
- Individual test case failures don't stop the evaluation
- Error details are logged for debugging
- Partial results are saved and can be resumed

### Log Analysis

Pipeline logs include:
- Search queries and results
- URL extraction and scraping
- Embedding and retrieval steps
- Context building process
- Final answer generation

Use these logs to debug pipeline behavior and optimize performance. 