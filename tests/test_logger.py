# Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import unittest
from .utils import safe_delete_file
import logging
from convAssist.logger import ConvAssistLogger
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
        self.file_path = "tests/test_data/"
        self.file_name = "test_log"
        self.full_file_path = Path(self.file_path) / (self.file_name + ".txt")
        # safe_delete_file(self.full_file_path)
        self.logger = ConvAssistLogger(self.file_name, self.file_path, logging.INFO)
            
    def tearDown(self) -> None: 
        #TODO: fix logger to not append a file extension...
        self.logger.Close()
        safe_delete_file(self.full_file_path)
        
           
        return super().tearDown()
    
    def testSetLogger(self):
        self.logger.setLogger()
        assert self.logger.IsLogInitialized()
        
    def testLogMessageAfterSetLogger(self):
        self.logger.setLogger()
        self.logger.info("All your bases are mine.")
        
        with open(self.full_file_path, 'r') as file:
            contents = file.read().strip()
            
        asssert_log_message(contents, "INFO", "All your bases are mine.")
        
    def testNoLogMessageWithoutSetLogger(self):
        self.logger.info("All your bases are mine.")
        
        assert Path(self.full_file_path).is_file() == False
        