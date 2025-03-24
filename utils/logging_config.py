import logging
from datetime import datetime
import os

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels"""
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'ENDC': '\033[0m',       # Reset color
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['ENDC']}"

def setup_logging():
    """Setup application logging with file and console handlers"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Generate log filename with timestamp
    log_filename = f"logs/graph_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup logger
    logger = logging.getLogger('graph_rag')
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler for more permanent logging
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# Get logger instance
logger = setup_logging() 