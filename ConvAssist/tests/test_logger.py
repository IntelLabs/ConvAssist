# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import sys
import unittest
from io import StringIO

from ConvAssist.utilities.logging_utility import LoggingUtility, QueueHandler


class TestLoggingUtility(unittest.TestCase):
    def setUp(self):
        self.logging_utility = LoggingUtility()

    def test_get_logger_with_stream_handler(self):
        logger_name = "test_logger"
        log_level = logging.DEBUG

        # Redirect stdout to capture log output
        captured_output = StringIO()
        sys.stdout = captured_output

        logger = self.logging_utility.get_logger(logger_name, log_level)

        # Test if logger has a stream handler
        self.assertTrue(
            any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
        )

        # Test if logger logs at the correct level
        logger.debug("This is a debug message")
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertIn("This is a debug message", captured_output.getvalue())

    def test_get_logger_with_queue_handler(self):
        logger_name = "test_logger_queue"
        log_level = logging.INFO

        logger = self.logging_utility.get_logger(logger_name, log_level, queue_handler=True)

        # Test if logger has a queue handler
        self.assertTrue(any(isinstance(handler, QueueHandler) for handler in logger.handlers))

        # Test if logger logs to the queue
        logger.info("This is an info message")
        log_record = self.logging_utility.central_log_queue.get()
        self.assertIn("This is an info message", log_record)


if __name__ == "__main__":
    unittest.main()
