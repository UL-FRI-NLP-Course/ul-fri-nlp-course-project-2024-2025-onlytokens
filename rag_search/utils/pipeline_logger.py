import json
from datetime import datetime
from typing import Any, Dict, List
import os

class PipelineLogger:
    """Logger class for the RAG pipeline that stores logs in memory and optionally saves to file."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger with a directory for log files.
        
        Args:
            log_dir: Directory where log files will be stored
        """
        self.log_dir = log_dir
        self._ensure_log_dir()
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"pipeline_log_{self.current_session}.jsonl")
        
        # In-memory storage for logs
        self.logs: List[Dict[str, Any]] = []
            
    def _ensure_log_dir(self):
        """Create the log directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def log(self, stage: str, data: Dict[str, Any], status: str = "info"):
        """Log pipeline stage information to memory.
        
        Args:
            stage: Name of the pipeline stage (e.g., "search", "scraping", etc.)
            data: Dictionary containing relevant data to log
            status: Status of the operation ("info", "error", "warning")
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "data": data
        }
        
        # Store in memory
        self.logs.append(log_entry)
            
    def log_error(self, stage: str, error: Exception, additional_data: Dict[str, Any] = None):
        """Log an error that occurred during pipeline execution.
        
        Args:
            stage: Name of the pipeline stage where error occurred
            error: The exception that was raised
            additional_data: Any additional context about the error
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            **(additional_data or {})
        }
        self.log(stage, error_data, status="error")
        
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs stored in memory.
        
        Returns:
            List of log entries
        """
        return self.logs
        
    def save_logs(self):
        """Save all logs to file."""
        with open(self.log_file, "w") as f:
            for log_entry in self.logs:
                f.write(json.dumps(log_entry) + "\n")
                
    def clear_logs(self):
        """Clear all logs from memory."""
        self.logs = [] 