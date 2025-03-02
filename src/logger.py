import logging
import time
from functools import wraps

from pythonjsonlogger import jsonlogger

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Sets up the logger to output in JSON format.
    
    :param log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
    :param log_file: Optional file path to log to a file. If None, logs to stdout.
    :return: A logger instance.
    """
    logger = logging.getLogger('json_logger')
    
    # Set the logging level
    logger.setLevel(log_level)

    # Create a formatter using the python-json-logger package
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s %(filename)s %(funcName)s %(lineno)s',
    )

    # Create the log handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()  # Logs to stdout by default

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_execution_time(logger):
    """
    A decorator to log execution time of a function with the given logger.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            # Log execution time with the provided logger
            logger.info(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator