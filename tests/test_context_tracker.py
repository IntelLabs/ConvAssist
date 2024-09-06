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
        self.my_callback = BufferedCallback("")

    def test_context_change(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.assertFalse(tracker.context_change())

    def test_prefix(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.my_callback.update("example prefix ")

        prefix = tracker.prefix()
        self.assertEqual(prefix, "")

    def test_token(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.my_callback.update("example token")
        token = tracker.token(0)
        self.assertEqual(token, "token")

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

    def test_is_completion_valid_token_exists(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.my_callback.update("example")
        completion = "example completion"
        self.assertTrue(tracker.is_completion_valid(completion))

    def test_is_completion_valid_token_not_exists(self):
        tracker = ContextTracker(self.config, self.my_callback)
        self.my_callback.update("missing")
        completion = "example completion"
        self.assertFalse(tracker.is_completion_valid(completion))
if __name__ == '__main__':
    unittest.main()