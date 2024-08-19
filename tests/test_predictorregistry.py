import unittest
from unittest.mock import mock_open, patch
import configparser

from src.predictor_registry import PredictorRegistry
from src.context_tracker import ContextTracker
from src.utilities.callback import Callback

class TestPredictorRegistry(unittest.TestCase):
    def setUp(self):
        self.config = configparser.ConfigParser()
        self.config['SpellCorrectPredictor'] = {
            'predictor_class': "SpellCorrectPredictor",
            'static_resources_path': './',
            'spellingdatabase': 'big.txt',
        }
        self.config['ContextTracker'] = {
            'sliding_window_size': '80',
            'lowercase_mode': 'True'
        }
        self.config['PredictorRegistry'] = {
            'predictors': ''
        }
        
        self.callback = Callback()
        self.predictor_registry = PredictorRegistry(self.config)
        self.context_tracker = ContextTracker(self.config, self.predictor_registry, self.callback)

    def test_set_predictors(self):
        self.predictor_registry.set_predictors()
        self.assertEqual(len(self.predictor_registry), 0)  # Check if the predictor list is empty

        # Add test cases to check if the predictors are added correctly based on the config

    def test_add_predictor(self):
        predictor_name = "SpellCorrectPredictor"  # Add a predictor name from the config
        self.predictor_registry.add_predictor(predictor_name)
        self.assertEqual(len(self.predictor_registry), 1)  # Check if the predictor is added correctly

        # Add test cases to check if the correct predictor is added based on the predictor name

    def test_model_status(self):
        # Add test cases to check the model status of each predictor in the registry
        pass
    
    # def test_close_database(self):
    #     self.predictor_registry.close_database()

        # Add test cases to check if the close_database() method is called for each predictor

if __name__ == '__main__':
    unittest.main()