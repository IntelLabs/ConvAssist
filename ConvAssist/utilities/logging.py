import logging
import os
import sys
from datetime import datetime
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

    @staticmethod
    def getFileName(name):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y___%H-%M-%S")
        return f"ConvAssist_Log{date_time}_{name}.log"
    
    def __init__(self, name, level="DEBUG", log_location=None):
        """
        Initializes the ConvAssistLogger object.

        Args:
            name (str, required): The name of the logger.
            level (str, optional): The log level. Defaults to "DEBUG".
            log_location (str, optional): The path for the log file. Defaults to None.
        """

        self.logger = logging.getLogger(f"ConvAssist_{name}")
        self.logger.propagate = False
        self.configure_logging(level, log_location)

    def configure_logging(self, level, log_location):
        loglevel = self._nameToLevel.get(level, logging.DEBUG)
        self.logger.setLevel(loglevel)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')

        if len(self.logger.handlers) > 0:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # add a file handler if log_location is provided
        if log_location:
            if not os.path.exists(log_location):
                os.makedirs(log_location)
            
            log_file = os.path.join(log_location, self.getFileName(self.logger.name))

            file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # add a stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def addHandler(self, handler):
        self.logger.addHandler(handler)
        
    def close(self):
        """Close all handlers associated with the logger."""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.debug("Closing handler: ", handler)
            handler.close()
            self.logger.removeHandler(handler)

    def debug(self, msg, exception=None):
        if exception:
            self.logger.debug(msg, exc_info=exception)
        else:
            self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg, exception=None):
        if exception:
            self.logger.critical(msg, exc_info=exception)
        else:
            self.logger.critical(msg)