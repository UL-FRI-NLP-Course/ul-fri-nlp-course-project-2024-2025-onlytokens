import os
import sys
import json
import csv
import yaml
import asyncio
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import debugpy
# Add parent directory to path to import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#set verbose to false
os.environ["VERBOSE"] = "False"

from main import (
    RAGSearchPipeline,
    load_config,
    get_api_config,
    OpenAIProvider,
    SearXNGProvider,
    WebScraper,
    Chunker,
    OpenAIEmbedder,
    CosineRetriever,
    JinaAIReranker,
    LukaContextBuilder,
    LLMQueryEnhancer,
    QualityImprover
)

import openai

class PipelineEvaluator:
    """Class to evaluate the RAG pipeline on a test set."""
    
    def __init__(
        self,
        config_path: str,
        test_set_path: str,
        results_dir: str,
        use_openai: bool = True,
        restore_run_id: str = None
    ):
        self.config_path = config_path
        self.test_set_path = test_set_path
        self.results_dir = Path(results_dir)
        self.use_openai = use_openai
        self.restore_run_id = restore_run_id
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if restore_run_id:
            # Load existing run metadata
            run_dir = self.results_dir / f"eval_run_{restore_run_id}"
            if not run_dir.exists():
                raise ValueError(f"No evaluation run found with ID {restore_run_id}")
            
            with open(run_dir / "run_metadata.json", 'r') as f:
                metadata = json.load(f)
            self.run_id = metadata['run_id']
            self.timestamp = metadata['timestamp']
        else:
            # Create new UUID and timestamp
            self.run_id = str(uuid.uuid4())
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize pipeline after run_id is created
        self.pipeline = self._initialize_pipeline()
        
    def _read_pipeline_logs(self) -> List[Dict[str, Any]]:
        """Get logs from the pipeline's in-memory storage."""
        return self.pipeline.logger.get_logs()

    def _initialize_pipeline(self) -> RAGSearchPipeline:
        """Initialize the RAG pipeline with the given configuration."""
        # Load configuration
        config = load_config(self.config_path)
        print(config)
        self.config = config

        # Set all verbose keys to false
        for key in config:
            for subkey in config[key]:
                if subkey.endswith("verbose"):
                    config[key][subkey] = False
        # Get unified API configuration
        api_config = get_api_config(config, self.use_openai)

        print("API CONFIG")
        print(api_config)
        print("CONFIG")
        print(config)
        # Initialize OpenAI clients
        embedding_client = openai.OpenAI(
            api_key=api_config["api_key"],
            base_url=api_config["embedding_url"]
        )
        llm_client = openai.OpenAI(
            api_key=api_config["api_key"],
            base_url=api_config["llm_url"]
        )

        # Initialize all pipeline components
        query_enhancer = LLMQueryEnhancer(
            openai_client=llm_client,
            model=api_config["llm_model"],
            max_queries=config["query_enhancer"]["max_queries"],
            verbose=config["query_enhancer"]["verbose"]
        )
        
        search_provider = SearXNGProvider(
            verbose=config["search_provider"]["verbose"],
            instance_url=config["search_provider"]["instance_url"]
        )
        
        quality_improver = QualityImprover(verbose=config["quality_improver"]["verbose"])
        
        web_scraper = WebScraper(
            strategies=config["web_scraper"]["strategies"],
            debug=config["web_scraper"]["debug"],
            llm_base_url=api_config["llm_url"],
            user_query=None,
            quality_improver=quality_improver,
            min_quality_score=config["quality_improver"]["min_quality_score"],
            enable_quality_model=config["quality_improver"]["enable_quality_model"],
        )

        chunker = Chunker(
            verbose=config["chunker"]["verbose"],
            chunk_size=config["chunker"]["chunk_size"],
            chunk_overlap=config["chunker"]["chunk_overlap"]
        )

        embedder = OpenAIEmbedder(
            openai_client=embedding_client,
            model_name=api_config["embedding_model"],
            verbose=config["embedder"]["verbose"],
            max_tokens=config["embedder"]["max_tokens"],
            batch_size=config["embedder"]["batch_size"]
        )
        
        retriever = CosineRetriever(
            embedder=embedder,
            verbose=config["retriever"]["verbose"],
            top_k=config["retriever"]["top_k"]
        )
        
        reranker = JinaAIReranker(
            verbose=config["reranker"]["verbose"],
            top_k=config["reranker"]["top_k"],
            batch_size=config["reranker"]["batch_size"],
            max_length=config["reranker"]["max_length"]
        )
        
        context_builder = LukaContextBuilder(verbose=config["context_builder"]["verbose"])
        
        llm_provider = OpenAIProvider(
            model=api_config["llm_model"],
            api_key=api_config["api_key"],
            api_base=api_config["llm_url"],
            verbose=config["llm_provider"]["verbose"]
        )

        # Initialize and return pipeline
        return RAGSearchPipeline(
            search_provider=search_provider,
            web_scraper=web_scraper,
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            context_builder=context_builder,
            llm_provider=llm_provider,
            query_enhancer=query_enhancer,
            max_sources=config["pipeline"]["max_sources"],
            debug=config["pipeline"]["debug"],
            log_dir=str(self.results_dir / f"pipeline_logs_{self.run_id}")
        )

    def load_test_set(self) -> List[Dict[str, Any]]:
        """Load test cases from CSV file."""
        test_cases = []
        with open(self.test_set_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse metadata string to dict
                try:
                    metadata = json.loads(row['metadata'].replace("'", '"'))
                except:
                    metadata = {}
                
                test_case = {
                    'metadata': metadata,
                    'question': row['question'],
                    'true_answer': row['true_answer']
                }
                test_cases.append(test_case)

        #keep to 5
        return test_cases

    async def evaluate_single_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case and return results."""
        try:
            # Clear pipeline conversation history and state
            self.pipeline.clear_conversation()
            
            # Clear any cached content in components
            self.pipeline.embedded_chunks = None
            self.pipeline.current_context = None
            
            # Clear previous logs
            self.pipeline.logger.clear_logs()
            
            # Temporarily disable auto-clearing of logs in the pipeline
            original_save_logs = self.pipeline.logger.save_logs
            original_clear_logs = self.pipeline.logger.clear_logs
            self.pipeline.logger.save_logs = lambda: None
            self.pipeline.logger.clear_logs = lambda: None
            
            try:
                # Run pipeline on test question
                response = await self.pipeline.run(test_case['question'])
                
                # Get logs from memory before restoring log clearing
                pipeline_logs = self.pipeline.logger.get_logs()
                
                # Extract relevant information from logs
                enhanced_queries = next((log['data']['enhanced_queries'] for log in pipeline_logs if log['stage'] == 'query_enhancement'), [])
                search_results = next((log['data'] for log in pipeline_logs if log['stage'] == 'search'), {})
                urls = next((log['data']['urls'] for log in pipeline_logs if log['stage'] == 'url_extraction'), [])
                
        
                
                result = {
                    'question': test_case['question'],
                    'true_answer': test_case['true_answer'],
                    'enhanced_queries': enhanced_queries,
                    'pipeline_answer': response,
                    'metadata': test_case['metadata'],
                    'retrieved_urls': urls,
                    'search_results': search_results,
                    'pipeline_logs': pipeline_logs,
                    'success': True,
                    'error': None,
                    'case_id': str(uuid.uuid4())  # Unique ID for each case
                }

                print("--------------------------------")
                print(f"Question: {test_case['question']} \nAnswer: {response}")
                print("--------------------------------")
            finally:
                # Restore original logger methods
                self.pipeline.logger.save_logs = original_save_logs
                self.pipeline.logger.clear_logs = original_clear_logs
                
                # Now save and clear logs
                self.pipeline.logger.save_logs()
                self.pipeline.logger.clear_logs()
            
        except Exception as e:
            result = {
                'question': test_case['question'],
                'true_answer': test_case['true_answer'],
                'pipeline_answer': None,
                'metadata': test_case['metadata'],
                'retrieved_urls': [],
                'search_results': {},
                'pipeline_logs': self.pipeline.logger.get_logs(),  # Get any logs that were generated before the error
                'success': False,
                'error': str(e),
                'case_id': str(uuid.uuid4())  # Unique ID for each case
            }
            
            # Save logs even in case of error
            self.pipeline.logger.save_logs()
            self.pipeline.logger.clear_logs()
            
        return result

    def _load_previous_results(self, results_file: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Load results from a previous run and get processed question IDs."""
        results = []
        processed_questions = set()
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    results.append(result)
                    processed_questions.add(result['question'])
                    
        return results, processed_questions

    async def run_evaluation(self):
        """Run evaluation on all test cases."""
        # Load test cases
        test_cases = self.load_test_set()
        
        # Create results for this run
        run_dir = self.results_dir / f"eval_run_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run metadata
        run_metadata = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config_path': str(self.config_path),
            'test_set_path': str(self.test_set_path),
            'use_openai': self.use_openai
        }
        
        # Save metadata and config if this is a new run
        if not self.restore_run_id:
            with open(run_dir / "run_metadata.json", 'w') as f:
                json.dump(run_metadata, f, indent=2)
            with open(run_dir / "config.yaml", 'w') as f:
                yaml.dump(self.config, f)
        
        # Create results file
        results_file = run_dir / "results.jsonl"
        
        # Load previous results if restoring
        results, processed_questions = self._load_previous_results(results_file)
        
        # Evaluate remaining test cases
        for i, test_case in enumerate(test_cases):
            # Skip already processed questions if restoring
            if test_case['question'] in processed_questions:
                print(f"Skipping already processed case {i+1}/{len(test_cases)}: {test_case['question']}")
                continue
                
            print(f"Evaluating case {i+1}/{len(test_cases)}: {test_case['question']}")
            result = await self.evaluate_single_case(test_case)
            results.append(result)
            
            # Append result to JSONL file
            with open(results_file, 'a') as f:
                json.dump({
                    'case_id': result['case_id'],
                    'question': result['question'],
                    'true_answer': result['true_answer'],
                    'pipeline_answer': result['pipeline_answer'],
                    'enhanced_queries': result['enhanced_queries'],
                    'metadata': result['metadata'],
                    'retrieved_urls': result['retrieved_urls'],
                    'search_results': result['search_results'],
                    'success': result['success'],
                    'error': result['error'],
                }, f)
                f.write('\n')
        
        # Save summary results
        summary = self._create_summary(results)
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from evaluation results."""
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r['success'])
        
        # Calculate average number of retrieved URLs
        avg_urls = sum(len(r['retrieved_urls']) for r in results) / total_cases if total_cases > 0 else 0
        
        # Collect all unique URLs
        all_urls = set()
        for r in results:
            all_urls.update(r['retrieved_urls'])
        
        summary = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'success_rate': successful_cases / total_cases if total_cases > 0 else 0,
            'average_urls_per_query': avg_urls,
            'unique_urls_total': len(all_urls),
            'config_path': str(self.config_path),
            'test_set_path': str(self.test_set_path),
            'results_directory': str(self.results_dir)
        }
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG Pipeline')
    parser.add_argument('--config', type=str, help='Path to pipeline configuration file', default='./config.yaml')
    parser.add_argument('--test-set', type=str, help='Path to test set CSV file', default='./simple_qa_test_set.csv')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory to store evaluation results')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI API instead of local endpoints')
    parser.add_argument('--restore', type=str, help='Run ID to restore and continue from', default=None)
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PipelineEvaluator(
        config_path=args.config,
        test_set_path=args.test_set,
        results_dir=args.results_dir,
        use_openai=False,
        restore_run_id=args.restore
    )
    
    # Run evaluation
    asyncio.run(evaluator.run_evaluation())

if __name__ == "__main__":
    #start debugpy
    debugpy.listen(5678)
    print("Waiting for client to attach...")
    debugpy.wait_for_client()
    main()
