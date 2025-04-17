from typing import Any, Dict, Optional
from abc import ABC

class BaseLogger(ABC):
    """Base logger class for RAG components."""
    
    def __init__(self, pipeline_logger=None, component_name: str = None):
        """Initialize the base logger.
        
        Args:
            pipeline_logger: Optional PipelineLogger instance to use
            component_name: Name of the component for logging
        """
        self.pipeline_logger = pipeline_logger
        self.component_name = component_name or self.__class__.__name__
        
    def log(self, operation: str, data: Dict[str, Any], status: str = "info"):
        """Log an operation with the pipeline logger if available.
        
        Args:
            operation: Name of the operation being performed
            data: Data to log
            status: Status of the operation
        """
        if self.pipeline_logger:
            stage = f"{self.component_name}.{operation}"
            self.pipeline_logger.log(stage, data, status)
            
    def log_error(self, operation: str, error: Exception, additional_data: Optional[Dict[str, Any]] = None):
        """Log an error with the pipeline logger if available.
        
        Args:
            operation: Name of the operation where error occurred
            error: The exception that was raised
            additional_data: Any additional context about the error
        """
        if self.pipeline_logger:
            stage = f"{self.component_name}.{operation}"
            self.pipeline_logger.log_error(stage, error, additional_data) 