# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base class for callbacks.

"""

from __future__ import absolute_import, unicode_literals

class BufferedCallback():

    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        
    def past_stream(self):
        return self.buffer
    
    def future_stream(self):
        return ""
    
    def update(self, text):
        self.buffer = text
