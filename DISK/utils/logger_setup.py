# logger_setup.py
import logging

# Create a logger instance
logger = logging.getLogger('DISKlogger')
logger.setLevel(logging.DEBUG)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)

# Set up basic configuration
logging.basicConfig(level=logging.DEBUG)
