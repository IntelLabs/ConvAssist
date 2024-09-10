import logging
import unittest
from unittest.mock import mock_open, patch
import configparser

from ConvAssist.predictor_registry import PredictorRegistry
from ConvAssist.context_tracker import ContextTracker

class TestPredictorRegistry(unittest.TestCase):
    def setUp(self):
        self.config = configparser.ConfigParser()
        #"PredictorRegistry", "predictors"
        self.config['PredictorRegistry'] = {
            'predictors': "SpellCorrectPredictor",
        }
        self.config['SpellCorrectPredictor'] = {
            'predictor_class': "SpellCorrectPredictor",
            'static_resources_path': './',
            'spellingdatabase': 'big.txt',
        }
        self.config['ContextTracker'] = {
            'sliding_window_size': '80',
            'lowercase_mode': 'True'
        }
        
        self.predictor_registry = PredictorRegistry()
        self.context_tracker = ContextTracker()

    def test_init(self):
        self.assertEqual(len(self.predictor_registry), 0)

    def test_set_predictors(self):
        self.predictor_registry.set_predictors(self.config, self.context_tracker, logging.getLogger())
        self.assertEqual(len(self.predictor_registry), 1)

    def test_get_predictor(self):
        self.predictor_registry.set_predictors(self.config, self.context_tracker, logging.getLogger())
        predictor = self.predictor_registry.get_predictor("SpellCorrectPredictor")
        self.assertIsNotNone(predictor)


if __name__ == '__main__':
    unittest.main()