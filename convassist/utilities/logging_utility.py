# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
import queue
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from concurrent_log_handler import ConcurrentRotatingFileHandler
from typing import TextIO

import colorlog

if sys.platform == "win32":
    import pydebugstring


class QueueHandler(logging.Handler):
    """
    This class is a custom logging handler that sends log messages to a queue.
    args:
        log_queue: queue.Queue - the queue to send log messages to
    returns:
        None
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))



class LoggingUtility:
    _instance = None
    _initialized = False


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingUtility, cls).__new__(cls)

            cls._central_log_queue = queue.Queue()
            cls._log_location = None
        return cls._instance

    def __init__(self):
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s %(message)s"
        self.date_format = "%m-%d-%Y %H:%M:%S"
        self._formatter: logging.Formatter = logging.Formatter(
            fmt=self.log_format, datefmt=self.date_format
        )
        self.file_handler = None


    @property
    def formatter(self):
        return self._formatter

    @property
    def central_log_queue(self):
        return self._instance._central_log_queue

    @staticmethod
    def getLogFileName():
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%dT%H_%M_%S")
        return f"ConvAssist_Log_{date_time}.txt"
    
    def set_log_location(self, log_location):
        if not os.path.exists(log_location):
            os.makedirs(log_location)

        self._instance._log_location = os.path.join(log_location, LoggingUtility.getLogFileName())
        self.file_handler = ConcurrentRotatingFileHandler(self._instance._log_location, maxBytes=1024 * 1024 * 3, backupCount=2)
        self.file_handler.setFormatter(self.formatter)


    def get_logger(self, name, log_level, log_file=True, queue_handler=False):

        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        logger.propagate = True

        logger.handlers.clear()

        # always add a stream handler
        self.add_stream_handler(logger, sys.stdout)

        # # # optionally add a file handler
        #TODO FIXME
        # if log_file:
        #     self.add_file_handler(logger)

        # optionally add a queue handler
        if queue_handler:
            self.add_queue_handler(logger)

        if sys.platform == "win32":
            # if log_level is logging.DEBUG add a pydebugstring handler
            logger.addHandler(pydebugstring.OutputDebugStringHandler())

        return logger

    def add_file_handler(self, logger: logging.Logger):
        #TODO FIXME
        if not self._instance._log_location:
            return

        if not self.file_handler:
            self.file_handler = self.set_log_location(self._instance._log_location)

        logger.addHandler(self.file_handler)

    def add_queue_handler(self, logger: logging.Logger):
        queue_handler = QueueHandler(self.central_log_queue)
        queue_handler.setFormatter(self.formatter)
        logger.addHandler(queue_handler)

    def add_stream_handler(self, logger: logging.Logger, textio: TextIO):
        stream_handler = logging.StreamHandler(textio)

        formatter = colorlog.ColoredFormatter(
            "%(asctime)s - %(name)s - %(log_color)s%(levelname)-8s%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            datefmt=self.date_format,
        )

        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
