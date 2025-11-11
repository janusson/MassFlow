def initialize_logging(log_file_path):
    """
    Initialize logging.
    """
    import logging
    from yogimass import yogimass_logging
    from datetime import datetime

    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info("Logging started at {datetime.now()}.")
    return log_formatter, logger


log_formatter, logger = yogimass_logging.main()

logger.info(f"{information}")
