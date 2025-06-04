"""
Log handler module for web UI.
"""
import logging
from typing import List, Dict
from datetime import datetime
import threading

class MemoryLogHandler(logging.Handler):
    """
    A logging handler that keeps logs in memory and provides access to them.
    Thread-safe implementation.
    """
    def __init__(self, capacity: int = 1000):
        """
        Initialize the handler with a maximum capacity.
        
        Args:
            capacity: Maximum number of log records to keep
        """
        super().__init__()
        self.capacity = capacity
        self.logs: List[Dict] = []
        self.last_access = {}  # Track last access by session_id
        self._lock = threading.Lock()
        
    def emit(self, record):
        """
        Add a log record to the memory buffer.
        
        Args:
            record: Log record to add
        """
        with self._lock:
            # Format the log record
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'message': self.format(record),
                'logger': record.name
            }
            
            # Add the log entry to the buffer
            self.logs.append(log_entry)
            
            # Remove oldest logs if we exceed capacity
            if len(self.logs) > self.capacity:
                self.logs = self.logs[-self.capacity:]
    
    def get_logs(self, session_id: str = None, since_index: int = 0) -> Dict:
        """
        Get logs since the given index.
        
        Args:
            session_id: Session ID for tracking last access
            since_index: Index from which to return logs
            
        Returns:
            Dict with logs and next index
        """
        with self._lock:
            # Get logs starting from since_index
            logs_to_return = self.logs[since_index:]
            
            # Track last access for this session
            if session_id is not None:
                self.last_access[session_id] = len(self.logs)
            
            return {
                'logs': logs_to_return,
                'next_index': len(self.logs)
            }
    
    def clear(self):
        """Clear all logs."""
        with self._lock:
            self.logs = []

# Create a global instance of the memory log handler
memory_log_handler = MemoryLogHandler(capacity=1000)

# Set up formatter for the handler
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
memory_log_handler.setFormatter(formatter)

# Function to add this handler to the root logger
def setup_memory_logging(level=logging.INFO):
    """
    Set up memory logging.
    
    Args:
        level: Logging level to use
    """
    # Add the memory handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_log_handler)
    
    # Make sure the handler only captures logs of the specified level or higher
    memory_log_handler.setLevel(level)
    
    # Don't change the root logger level as it might affect other handlers
    
    logging.info("Memory logging set up successfully")
