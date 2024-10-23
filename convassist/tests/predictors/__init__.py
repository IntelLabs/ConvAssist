import configparser
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.sentence_completion_predictor import (
    SentenceCompletionPredictor,
)

from .. import setup_utils

class TestPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_utils.copy_static_resources()

    @classmethod
    def tearDownClass(cls):
        setup_utils.remove_static_resources()

