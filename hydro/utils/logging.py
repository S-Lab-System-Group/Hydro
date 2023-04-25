import logging



def logger_init(file):
    logger = logging.getLogger()
    logging.getLogger().handlers.clear()
    logger.setLevel(logging.INFO)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%b %d %H:%M:%S")
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)

    if file is not None:
        handler_file = logging.FileHandler(f"{file}.log", "w")
        handler_file.setLevel(logging.INFO)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    return logger
