import logging
import os


class ImportantLoggerAdapter(logging.LoggerAdapter):
    def important(self, msg, *args, **kwargs):
        kwargs["stacklevel"] = 2

        # Add our custom 'important' flag to the 'extra' dict for the filter.
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["important"] = True

        # Call the standard 'log' method, which correctly handles kwargs.
        self.logger.log(logging.INFO, msg, *args, **kwargs)

    def process(self, msg, kwargs):
        return msg, kwargs


class ImportantInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING or getattr(record, "important", False)


def setup_logger():
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%H:%M:%S",
    )

    # file handler
    file_handler = logging.FileHandler("logs/sekai_system.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ImportantInfoFilter())

    logger = logging.getLogger("sekai")
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handler addition
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return ImportantLoggerAdapter(logger, {})