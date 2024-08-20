import logging
import sys
from logging.handlers import RotatingFileHandler

class ConvAssistLogger:
    # Define logging levels as class attributes
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    _nameToLevel = {
        'CRITICAL': CRITICAL,
        'ERROR': ERROR,
        'WARNING': WARNING,
        'INFO': INFO,
        'DEBUG': DEBUG,
        'NOTSET': NOTSET
    }

    def __init__(self, name='ConvAssistLogger', level="DEBUG", log_file=None):
        self.logger = logging.getLogger(name)
        loglevel = self._nameToLevel.get(level, logging.DEBUG)
        self.logger.setLevel(loglevel)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')

        if len(self.logger.handlers) > 0:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        if log_file:
            file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def close(self):
        """Close all handlers associated with the logger."""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.debug("Closing handler: ", handler)
            handler.close()
            self.logger.removeHandler(handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)