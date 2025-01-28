# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
import queue
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
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

            cls._central_log_queue = queue.LifoQueue()
        return cls._instance

    def __init__(self):
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s %(message)s"
        self.date_format = "%m-%d-%Y %H:%M:%S"
        self._formatter: logging.Formatter = logging.Formatter(
            fmt=self.log_format, datefmt=self.date_format
        )

    @property
    def formatter(self):
        return self._formatter

    @property
    def central_log_queue(self):
        return self._instance._central_log_queue

    @staticmethod
    def getLogFileName(name):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y___%H-%M-%S")
        return f"ConvAssist_Log{date_time}_{name}.log"

    def get_logger(self, name, log_level, log_location=None, queue_handler=False):

        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        logger.propagate = True

        logger.handlers.clear()

        # always add a stream handler
        self.add_stream_handler(logger, sys.stdout)

        # # optionally add a file handler
        if log_location:
            self.add_file_handler(logger, log_location)

        # optionally add a queue handler
        if queue_handler:
            self.add_queue_handler(logger)

        if sys.platform == "win32":
            # if log_level is logging.DEBUG add a pydebugstring handler
            logger.addHandler(pydebugstring.OutputDebugStringHandler())

        return logger

    def add_file_handler(self, logger: logging.Logger, log_location: str):
        if not os.path.exists(log_location):
            os.makedirs(log_location)

        log_file = os.path.join(log_location, LoggingUtility.getLogFileName(logger.name))

        file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=2)
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)

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
