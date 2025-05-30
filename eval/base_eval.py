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

# Add parent directory to path to import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set verbose to false
os.environ["VERBOSE"] = "False"

from main import (
    load_config,
    get_api_config,
    OpenAIProvider
)

import openai

class BaseLLMEvaluator:
    """Class to evaluate direct LLM responses without RAG pipeline."""
    
    def __init__(
        self,
        config_path: str,
        test_set_path: str,
        results_dir: str,
        use_openai: bool = True,
        restore_run_id: Optional[str] = None,
        max_concurrent_requests: int = 5
    ):
        self.config_path = config_path
        self.test_set_path = test_set_path
        self.results_dir = Path(results_dir)
        self.use_openai = use_openai
        self.restore_run_id = restore_run_id
        self.config: Dict[str, Any] = {}  # Initialize config attribute
        self.max_concurrent_requests = max_concurrent_requests
        
        # Create semaphore for limiting concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if restore_run_id:
            # Load existing run metadata
            run_dir = self.results_dir / f"base_eval_run_{restore_run_id}"
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
        
        # Initialize LLM after run_id is created
        self.llm_provider = self._initialize_llm()
        
    def _initialize_llm(self) -> OpenAIProvider:
        """Initialize the LLM provider with the given configuration."""
        # Load configuration
        config = load_config(self.config_path)
        print("Loaded config:", config)
        self.config = config

        # Get unified API configuration
        api_config = get_api_config(config, self.use_openai)
        print("API CONFIG:", api_config)

        # Initialize LLM provider
        llm_provider = OpenAIProvider(
            model=api_config["llm_model"],
            api_key=api_config["api_key"],
            api_base=api_config["llm_url"],
            verbose=False  # Set to False for cleaner evaluation output
        )

        return llm_provider

    def load_test_set(self) -> List[Dict[str, Any]]:
        """Load test cases from CSV file."""
        test_cases = []
        
        # Try different encodings to handle Unicode issues
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(self.test_set_path, 'r', encoding=encoding) as f:
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
                
                print(f"Successfully loaded {len(test_cases)} test cases using {encoding} encoding")
                break  # If successful, break out of the loop
                
            except UnicodeDecodeError as e:
                print(f"Failed to read with {encoding} encoding: {e}")
                test_cases = []  # Reset for next attempt
                continue
            except Exception as e:
                print(f"Error reading file with {encoding} encoding: {e}")
                test_cases = []  # Reset for next attempt
                continue
        
        if not test_cases:
            raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings_to_try}")

        return test_cases

    async def evaluate_single_case_with_save(self, test_case: Dict[str, Any], case_index: int, total_cases: int, results_file: Path) -> Dict[str, Any]:
        """Evaluate a single test case and immediately save the result."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                print(f"Starting evaluation {case_index+1}/{total_cases}: {test_case['question'][:50]}...")
                
                # Create messages for direct LLM query
                messages = [
                    {"role": "user", "content": test_case['question']}
                ]
                
                # Ask LLM the question directly
                response = await self.llm_provider.generate(messages)
                
                result = {
                    'question': test_case['question'],
                    'true_answer': test_case['true_answer'],
                    'llm_answer': response,
                    'metadata': test_case['metadata'],
                    'success': True,
                    'error': None,
                    'case_id': str(uuid.uuid4()),  # Unique ID for each case
                    'original_index': case_index  # Track original order
                }

                print(f"Completed evaluation {case_index+1}/{total_cases}")
                print("--------------------------------")
                print(f"Question: {test_case['question']} \nLLM Answer: {response}")
                print("--------------------------------")
                
                # Save result immediately
                await self._save_single_result(result, results_file)
                    
            except Exception as e:
                result = {
                    'question': test_case['question'],
                    'true_answer': test_case['true_answer'],
                    'llm_answer': None,
                    'metadata': test_case['metadata'],
                    'success': False,
                    'error': str(e),
                    'case_id': str(uuid.uuid4()),  # Unique ID for each case
                    'original_index': case_index  # Track original order
                }
                print(f"Error in evaluation {case_index+1}/{total_cases}: {str(e)}")
                
                # Save error result immediately
                await self._save_single_result(result, results_file)
                
            return result

    async def _save_single_result(self, result: Dict[str, Any], results_file: Path):
        """Save a single result to the results file with file locking."""
        import asyncio
        
        # Use a file lock to prevent concurrent writes from corrupting the file
        lock_file = results_file.with_suffix('.lock')
        
        # Wait for lock to be available
        while lock_file.exists():
            await asyncio.sleep(0.01)
        
        try:
            # Create lock
            lock_file.touch()
            
            # Append result to JSONL file
            with open(results_file, 'a') as f:
                json.dump({
                    'case_id': result['case_id'],
                    'question': result['question'],
                    'true_answer': result['true_answer'],
                    'llm_answer': result['llm_answer'],
                    'metadata': result['metadata'],
                    'success': result['success'],
                    'error': result['error'],
                    'original_index': result['original_index']
                }, f)
                f.write('\n')
                f.flush()  # Ensure immediate write to disk
                
        finally:
            # Remove lock
            if lock_file.exists():
                lock_file.unlink()

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
        """Run evaluation on all test cases with concurrent processing and real-time saving."""
        # Load test cases
        test_cases = self.load_test_set()
        
        # Create results for this run
        run_dir = self.results_dir / f"base_eval_run_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run metadata
        run_metadata = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config_path': str(self.config_path),
            'test_set_path': str(self.test_set_path),
            'use_openai': self.use_openai,
            'evaluation_type': 'base_llm',
            'max_concurrent_requests': self.max_concurrent_requests
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
        existing_results, processed_questions = self._load_previous_results(results_file)
        
        # Filter out already processed test cases and keep track of their original indices
        remaining_test_cases = []
        for i, test_case in enumerate(test_cases):
            if test_case['question'] not in processed_questions:
                remaining_test_cases.append((i, test_case))
        
        if not remaining_test_cases:
            print("All test cases have already been processed!")
            return self._create_summary(existing_results)
        
        print(f"Processing {len(remaining_test_cases)} remaining test cases with max {self.max_concurrent_requests} concurrent requests...")
        print(f"Results will be saved in real-time to: {results_file}")
        
        # Create tasks for concurrent evaluation that save results immediately
        tasks = []
        for original_index, test_case in remaining_test_cases:
            task = self.evaluate_single_case_with_save(test_case, original_index, len(test_cases), results_file)
            tasks.append(task)
        
        # Run all tasks concurrently
        start_time = datetime.now()
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Process any exceptions that occurred
        successful_results = []
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception):
                print(f"Exception in task {i}: {result}")
            elif isinstance(result, dict):
                successful_results.append(result)
        
        # Load all results from file (including what was just written)
        all_results_from_file = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        all_results_from_file.append(result)
        
        # Sort results by original index to maintain input file order
        all_results_from_file.sort(key=lambda x: x.get('original_index', 0))
        
        # Calculate and print timing information
        duration = end_time - start_time
        avg_time_per_request = duration.total_seconds() / len(remaining_test_cases)
        
        print(f"\nEvaluation completed!")
        print(f"Total time: {duration}")
        print(f"Average time per request: {avg_time_per_request:.2f} seconds")
        print(f"Processed {len(remaining_test_cases)} test cases with {self.max_concurrent_requests} concurrent requests")
        print(f"Results saved to: {results_file}")
        
        # Save summary results
        summary = self._create_summary(all_results_from_file)
        summary['evaluation_duration_seconds'] = duration.total_seconds()
        summary['average_time_per_request_seconds'] = avg_time_per_request
        summary['concurrent_requests_used'] = self.max_concurrent_requests
        
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from evaluation results."""
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r['success'])
        
        summary = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'evaluation_type': 'base_llm',
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'success_rate': successful_cases / total_cases if total_cases > 0 else 0,
            'config_path': str(self.config_path),
            'test_set_path': str(self.test_set_path),
            'results_directory': str(self.results_dir)
        }
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Base LLM (without RAG)')
    parser.add_argument('--config', type=str, help='Path to pipeline configuration file', default='./config.yaml')
    parser.add_argument('--test-set', type=str, help='Path to test set CSV file', default='./simple_qa_test_set.csv')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory to store evaluation results')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI API instead of local endpoints')
    parser.add_argument('--restore', type=str, help='Run ID to restore and continue from', default=None)
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum number of concurrent requests (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BaseLLMEvaluator(
        config_path=args.config,
        test_set_path=args.test_set,
        results_dir=args.results_dir,
        use_openai=False,
        restore_run_id=args.restore,
        max_concurrent_requests=args.max_concurrent
    )
    
    # Run evaluation
    summary = asyncio.run(evaluator.run_evaluation())
    print("Evaluation completed!")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main() 