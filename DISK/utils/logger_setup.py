# logger_setup.py
import logging
import os
from omegaconf import DictConfig, OmegaConf

def setup_custom_logging(log_directory, log_filename, flag=logging.INFO):
    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Create a formatter to define the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_file_path = os.path.join(log_directory, log_filename)

    logger = logging.getLogger('DISK')

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(flag)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(flag)  # You can set the desired log level for console output
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def copy_config_file(modif_config, outputfile):
    with open(outputfile, 'w') as file:
        OmegaConf.save(modif_config, file)