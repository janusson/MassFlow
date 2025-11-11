def get_logger(name: str = "yogimass"):
    """
    Minimal and consistent logger for Yogimass modules.
    Usage: logger = get_logger(__name__)
    """
    import logging
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger
