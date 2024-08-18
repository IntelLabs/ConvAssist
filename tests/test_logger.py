# Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import unittest
# import io
import re
# import contextlib
# import sys 

from .utils import safe_delete_file
from ConvAssist.utilities.logging import ConvAssistLogger
from pathlib import Path

def asssert_log_message(log_string, log_level, expected):
    escaped_log_level = re.escape(log_level)
    escaped_exepected = re.escape(expected)
    pattern = rf"{escaped_log_level} {escaped_exepected}"
    
    match = re.search(pattern, log_string)
    
    # Assert that the pattern was found in the log string
    assert match is not None, f"Log string does not contain '{log_level}' followed by {expected}."

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.full_file_path = Path( "tests/test_data/") / ("test_log.log")
        self.my_logger = None            
    def tearDown(self) -> None: 
        if self.my_logger:
            self.my_logger.close()
        safe_delete_file(self.full_file_path)
                   
        return super().tearDown()
        
    def testInfo(self):
        self.my_logger = ConvAssistLogger(level="INFO", log_file=self.full_file_path)

        self.my_logger.info("All your bases are mine.")
        
        with open(self.full_file_path, 'r') as file:
            contents = file.read().strip()
            
        asssert_log_message(contents, "INFO", "All your bases are mine.")
    
    def testClose(self):
        self.my_logger = ConvAssistLogger(level="INFO", log_file=self.full_file_path)

        self.my_logger.close()
        self.my_logger.info("All your bases are mine.")
        
        with open(self.full_file_path, 'r') as file:
            contents = file.read().strip()
            
        assert contents == "", "Log file was not closed."
        