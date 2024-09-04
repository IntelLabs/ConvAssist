import logging
import os
import sys
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# class ConvAssistLogger():

#     def __init__(self, name):
#         self.name = name
#         self.logger = logging.getLogger(name)

    
#     def configure_logging(self, level=logging.NOTSET, log_location=None):
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')

#         if not level == logging.NOTSET:
#             self.logger.setLevel(level)

#         # if len(self.handlers) > 0:
#         #     for handler in self.handlers[:]:
#         #         self.removeHandler(handler)

#         # add a file handler if log_location is provided
#         if log_location:
#             if not os.path.exists(log_location):
#                 os.makedirs(log_location)
            
#             log_file = os.path.join(log_location, self.getLogFileName(self.name))

#             file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
#             file_handler.setFormatter(formatter)
#             self.logger.addHandler(file_handler)

#         # add a stream handler
#         stream_handler = logging.StreamHandler(sys.stdout)
#         stream_handler.setFormatter(formatter)
#         self.logger.addHandler(stream_handler)

#     def getLogger(self):
#         return self.logger

# logger_utility.py
import logging
import os
from typing import TextIO

class LoggingUtility:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')
    queue_handler = None
    @staticmethod
    def getLogFileName(name):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y___%H-%M-%S")
        return f"ConvAssist_Log{date_time}_{name}.log"

    @staticmethod
    def get_logger(name, log_level, log_location=None, queue_handler=None):
        # create a formatter
        LoggingUtility.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')
        if queue_handler:
            LoggingUtility.queue_handler = queue_handler

        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = False

        # always add a stream handler
        LoggingUtility.add_stream_handler(logger, sys.stdout)

        # optionally add a file handler
        if log_location: LoggingUtility.add_file_handler(logger, log_location)

        # optionally add a queue handler
        if queue_handler: LoggingUtility.add_queue_handler(logger, queue_handler)

        return logger

    @staticmethod
    def add_file_handler(logger:logging.Logger, log_location:str):
        if not os.path.exists(log_location):
            os.makedirs(log_location)

        log_file = os.path.join(log_location, LoggingUtility.getLogFileName(logger.name))

        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
        file_handler.setFormatter(LoggingUtility.formatter)
        logger.addHandler(file_handler)

    @staticmethod
    def add_queue_handler(logger:logging.Logger, queue_handler:logging.Handler):
        #queue_handler should already have a formatter set!
        if not queue_handler.formatter:
            queue_handler.setFormatter(LoggingUtility.formatter)
        logger.addHandler(queue_handler)

    @staticmethod
    def add_stream_handler(logger:logging.Logger, textio: TextIO):
        stream_handler = logging.StreamHandler(textio)
        stream_handler.setFormatter(LoggingUtility.formatter)
        logger.addHandler(stream_handler)