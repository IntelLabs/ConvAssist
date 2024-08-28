import unittest
from configparser import ConfigParser
from unittest.mock import MagicMock
from ConvAssist.context_tracker import ContextTracker, InvalidCallbackException
from ConvAssist.utilities.callback import BufferedCallback
from ConvAssist.predictor_registry import PredictorRegistry

class TestContextTracker(unittest.TestCase):
    def setUp(self):
        self.config:ConfigParser = ConfigParser()
        self.config['ContextTracker'] = {
            'sliding_window_size': '80',
            'lowercase_mode': 'True'
        }
        # self.config['PredictorRegistry'] = {
        #     'predictors': ''
        # }

        # self.predictor_registry = PredictorRegistry(self.config)
        self.my_callback = BufferedCallback("")

    def test_context_change(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.assertFalse(tracker.context_change())

    # def test_update_context(self):
    #     tracker = ContextTracker(self.config, self.my_callback)
    #     tracker.update_context()
    #     # Add assertions here

    # def test_prefix(self):
    #     tracker = ContextTracker(self.config, self.my_callback)
    #     prefix = tracker.prefix()
    #     # Add assertions here

    # def test_token(self):
    #     tracker = ContextTracker(self.config, self.my_callback)
    #     token = tracker.token(0)
    #     # Add assertions here

    # def test_extra_token_to_learn(self):
    #     tracker = ContextTracker(self.config, self.my_callback)
    #     extra_token = tracker.extra_token_to_learn(0, "change")
    #     # Add assertions here

    def test_future_stream(self):
        BufferedCallback.future_stream = MagicMock(return_value=(""))
        tracker = ContextTracker(self.config, self.my_callback)
        future_stream = tracker.future_stream()
        assert future_stream == ""
        BufferedCallback.future_stream.assert_called_once()

    def test_past_stream(self):
        BufferedCallback.past_stream = MagicMock(return_value=(""))
        tracker = ContextTracker(self.config, self.my_callback)
        past_stream = tracker.past_stream()
        assert past_stream == ""
        BufferedCallback.past_stream.assert_called_once()

    def test_is_completion_valid(self):
        tracker = ContextTracker(self.config, self.my_callback)
        completion = "example completion"
        self.assertFalse(tracker.is_completion_valid(completion))

if __name__ == '__main__':
    unittest.main()