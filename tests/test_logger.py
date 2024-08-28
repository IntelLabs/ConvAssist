import unittest
import os
import shutil
from unittest.mock import patch
import logging
from logging import handlers, CRITICAL, DEBUG, ERROR, INFO, WARNING
from ConvAssist.utilities.logging import ConvAssistLogger

current_location = os.path.dirname(os.path.abspath(__file__))

class TestConvAssistLogger(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists(os.path.join(current_location, "logs")):
            shutil.rmtree(os.path.join(current_location, "logs"))

        self.logger: ConvAssistLogger | None = None
        
        return super().setUp()

    def tearDown(self) -> None:
        if self.logger:
            self.logger.close()

        if os.path.exists(os.path.join(current_location, "logs")):
            shutil.rmtree(os.path.join(current_location, "logs"))
        
        return super().tearDown()

    def test_configure_logging_with_log_location(self):
        self.logger = ConvAssistLogger("test", level="DEBUG", log_location=os.path.join(current_location, "logs"))
        self.assertEqual(len(self.logger.logger.handlers), 2)
        self.assertIsInstance(self.logger.logger.handlers[0], handlers.RotatingFileHandler)
        self.assertIsInstance(self.logger.logger.handlers[1], logging.StreamHandler)

    def test_configure_logging_without_log_location(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        self.assertEqual(len(self.logger.logger.handlers), 1)
        self.assertIsInstance(self.logger.logger.handlers[0], logging.StreamHandler)

    def test_add_handler(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        self.assertIn(handler, self.logger.logger.handlers)

    def test_close(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        self.logger.close()
        self.assertNotIn(handler, self.logger.logger.handlers)

    def test_debug_without_exception(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        with patch.object(self.logger.logger, "debug") as mock_debug:
            self.logger.debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")

    def test_debug_with_exception(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        exception = Exception("Test exception")
        with patch.object(self.logger.logger, "debug") as mock_debug:
            self.logger.debug("Debug message", exception=exception)
            mock_debug.assert_called_once_with("Debug message", exc_info=exception)

    def test_info(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        with patch.object(self.logger.logger, "info") as mock_info:
            self.logger.info("Info message")
            mock_info.assert_called_once_with("Info message")

    def test_warning(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        with patch.object(self.logger.logger, "warning") as mock_warning:
            self.logger.warning("Warning message")
            mock_warning.assert_called_once_with("Warning message")

    def test_error(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        with patch.object(self.logger.logger, "error") as mock_error:
            self.logger.error("Error message")
            mock_error.assert_called_once_with("Error message")

    def test_critical_without_exception(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        with patch.object(self.logger.logger, "critical") as mock_critical:
            self.logger.critical("Critical message")
            mock_critical.assert_called_once_with("Critical message")

    def test_critical_with_exception(self):
        self.logger = ConvAssistLogger("test", level="DEBUG")
        exception = Exception("Test exception")
        with patch.object(self.logger.logger, "critical") as mock_critical:
            self.logger.critical("Critical message", exception=exception)
            mock_critical.assert_called_once_with("Critical message", exc_info=exception)

if __name__ == "__main__":
    unittest.main()