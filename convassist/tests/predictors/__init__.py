# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# import configparser
import unittest

from convassist.tests import setup_utils

# from unittest.mock import patch

# from parameterized import parameterized

# from convassist.context_tracker import ContextTracker
# from convassist.predictor.sentence_completion_predictor import (
#     SentenceCompletionPredictor,
# )


class TestPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_utils.copy_static_resources()

    @classmethod
    def tearDownClass(cls):
        setup_utils.remove_static_resources()
