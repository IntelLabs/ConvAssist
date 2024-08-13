# Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import unittest
from .utils import safe_delete_file
from convAssist import ConvAssistLogger
from pathlib import Path
import re

def asssert_log_message(log_string, log_level, expected):
    escaped_log_level = re.escape(log_level)
    escaped_exepected = re.escape(expected)
    pattern = rf"{escaped_log_level} {expected}"
    
    match = re.search(pattern, log_string)
    
    # Assert that the pattern was found in the log string
    assert match is not None, f"Log string does not contain '{log_level}' followed by {expected}."

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.full_file_path = Path( "tests/test_data/") / ("test_log.log")
        self.my_logger = ConvAssistLogger(log_level=ConvAssistLogger.INFO, log_file_name=self.full_file_path)
            
    def tearDown(self) -> None: 
        self.my_logger.close()
        safe_delete_file(self.full_file_path)
                   
        return super().tearDown()

        
    def testInfo(self):
        self.my_logger.info("All your bases are mine.")
        
        with open(self.full_file_path, 'r') as file:
            contents = file.read().strip()
            
        asssert_log_message(contents, "INFO", "All your bases are mine.")
                