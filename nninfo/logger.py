import logging


def get_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s: %(message)s")

    c_handler.setFormatter(c_format)

    c_handler.setLevel(logging.WARNING)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    return logger


def add_exp_file_handler(experiment_dir):
    main_logger = logging.getLogger("nninfo")
    f_handler = logging.FileHandler(experiment_dir / "log" / "exp_log.log")
    f_format = logging.Formatter(
        "%(asctime)s [%(name)-13.13s] [%(levelname)-5.5s]  %(message)s"
    )
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.INFO)
    # check if logger for this experiment exists already
    if main_logger.handlers:
        for h in main_logger.handlers:
            if h.__dict__["baseFilename"] == f_handler.baseFilename:
                return
    main_logger.addHandler(f_handler)


def remove_exp_file_handler(experiment_dir):
    main_logger = logging.getLogger("nninfo")
    dummy = logging.FileHandler(experiment_dir + "log/exp_log.log")
    for h in main_logger.handlers:
        if h.__dict__["baseFilename"] == dummy.baseFilename:
            main_logger.handlers.remove(h)
