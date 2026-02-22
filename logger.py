import logging
import os
from datetime import datetime

# Create log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Ensure logs directory exists
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Log format
LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# Create logger
logger = logging.getLogger("SupplyChainAI")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger to avoid duplicates

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Add handlers only if not already present
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
