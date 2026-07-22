import logging
import sys
from dataset.configs.settings import config

def get_dataset_logger(name: str = "vajra.dataset") -> logging.Logger:
    """
    Returns a configured logger for the dataset collection framework.
    Ensures consistent formatting and log levels across the dataset system.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    return logger

logger = get_dataset_logger()
