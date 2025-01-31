import unittest
from unittest.mock import patch

from convassist.tests import setup_utils

class TestPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_utils.copy_static_resources()

    @classmethod
    def tearDownClass(cls):
        setup_utils.remove_static_resources()

