# logger_setup.py
import logging
import os

import yaml


class VoidHandler(logging.Handler):
    def emit(self, record):
        pass  # Discard all logs

def setup_custom_logging(log_directory, log_filename, flag=logging.INFO):
    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Create a formatter to define the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_file_path = os.path.join(log_directory, log_filename)

    logger = logging.getLogger('DISK')

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)

    if not logger.handlers:  # Only add handlers if none exist

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(flag)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(flag)  # You can set the desired log level for console output
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for handler in logger.handlers:
        handler.flush()

    logger.info('Hey')

    return logger


def copy_config_file(modif_config, outputfile):
    with open(outputfile, 'w') as file:
        yaml.safe_dump(modif_config, file)