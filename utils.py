import os
import json
import logging


def save_to_json(objects_dict, output_dir_path):
    for name, obj in objects_dict.items():
        save_path = os.path.join(output_dir_path, f"{name}.json")

        with open(save_path, 'w') as f:
            json.dump(obj, f, indent=4)


def setup_logger(log_path, to_console=False, logger_name=__name__):
    # Use the provided logger_name or default to __name__
    logger = logging.getLogger(logger_name)

    # Avoid configuring multiple times by checking if handlers exist
    if not logger.handlers:  # Check if any handlers are already added
        # Set the default log level
        logger.setLevel(logging.DEBUG)

        # File handler (only if log_path is provided)
        if not to_console:
            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s %(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]"
            ))
            logger.addHandler(file_handler)

        # Console handler if requested
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s %(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]"
            ))
            logger.addHandler(console_handler)

    return logger