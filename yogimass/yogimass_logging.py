#!/usr/bin/python
# Standard logging setup for Yogimass
# ⌬ Logging setup

import logging

logging.basicConfig(level=logging.DEBUG)

def default_logger_config():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s:%(name)s: %(message)s")
    return log_formatter, logger

def error_logger(formatter):
    file_handler = logging.FileHandler(f"log/{__name__}.log")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    return file_handler


def info_logger(handler, formatter):
    stream_handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    return stream_handler


def main():
    log_formatter, logger = default_logger_config()
    file_handler = error_logger(log_formatter)
    stream_handler = info_logger(file_handler, log_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return log_formatter, logger


if __name__ == "__main__":
    main()
