import unittest
from configparser import ConfigParser
from ConvAssist.context_tracker import ContextTracker, InvalidCallbackException
from ConvAssist.utilities.callback import Callback
from ConvAssist.predictor_registry import PredictorRegistry

class TestContextTracker(unittest.TestCase):
    def setUp(self):
        self.config:ConfigParser = ConfigParser()
        self.config['ContextTracker'] = {
            'sliding_window_size': '80',
            'lowercase_mode': 'True'
        }
        self.config['PredictorRegistry'] = {
            'predictors': ''
        }

        self.predictor_registry = PredictorRegistry(self.config)
        self.my_callback = Callback()

    def test_context_change(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        self.assertFalse(tracker.context_change())

    def test_update_context(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        tracker.update_context()
        # Add assertions here

    def test_prefix(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        prefix = tracker.prefix()
        # Add assertions here

    def test_token(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        token = tracker.token(0)
        # Add assertions here

    def test_extra_token_to_learn(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        extra_token = tracker.extra_token_to_learn(0, "change")
        # Add assertions here

    def test_future_stream(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        future_stream = tracker.future_stream()
        # Add assertions here

    def test_past_stream(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        past_stream = tracker.past_stream()
        # Add assertions here

    def test_is_completion_valid(self):
        tracker = ContextTracker(self.config, self.predictor_registry, self.my_callback)
        completion = "example completion"
        self.assertFalse(tracker.is_completion_valid(completion))

    def test_invalid_callback_exception(self):
        with self.assertRaises(InvalidCallbackException):
            ContextTracker(self.config, self.predictor_registry, None)

if __name__ == '__main__':
    unittest.main()