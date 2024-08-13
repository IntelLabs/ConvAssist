# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import logging
import sys
from logging.handlers import RotatingFileHandler


class ConvAssistLogger:
    # Define logging levels as class attributes
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    _levelToName = {
        CRITICAL: 'CRITICAL',
        ERROR: 'ERROR',
        WARNING: 'WARNING',
        INFO: 'INFO',
        DEBUG: 'DEBUG',
        NOTSET: 'NOTSET'
    }
    
    def __init__(self, log_to_file=True, log_level=logging.INFO, log_file_name="convAssist"):
        """
            Initialize the logger with the given parameters

        Args:
            log_to_file (bool): Log to file or to console
            log_level (int): Log level
            log_file_name (str): Name of the log file

        Returns:
            ConvAssistLogger: Instance of the logger
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        if log_to_file:
            handler = logging.FileHandler(f"{log_file_name}")
        else:
            handler = logging.StreamHandler(sys.stdout)
            
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(log_formatter)
        
        self.logger.addHandler(handler)

    def close(self):
        """
        Clears and closes all handlers used to log into the file
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
        self.isLogInitialized = False

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
