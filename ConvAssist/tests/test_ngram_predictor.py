import unittest
from unittest.mock import MagicMock, patch
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from ConvAssist.utilities.suggestion import Suggestion
from ConvAssist.predictor.utilities.prediction import Prediction

class TestSmoothedNgramPredictor(unittest.TestCase):
    
    def setUp(self):
        self.config = MagicMock()
        self.context_tracker = ContextTracker()
        self.context_tracker.context = "Hello World"
        self.context_tracker.get_tokens.return_value = 0

        self.predictor_name = "GeneralWord"
        self.predictor.name = "GeneralWord"
        self.predictor.startwords = "startwords.json"
        patch("builtins.open", unittest.mock.mock_open(read_data='{"hello": 0.5, "world": 0.5}'))

        self.predictor = SmoothedNgramPredictor(
                self.config, self.context_tracker, self.predictor_name
            )
        self.predictor.db = MagicMock()
        self.predictor.logger = MagicMock()
        self.predictor.deltas = [0.5, 0.3, 0.2]
        self.predictor.cardinality = 3

    def test_predict_no_tokens(self):
        self.context_tracker.get_tokens.return_value = 0
        self.predictor.name = "GeneralWord"
        self.predictor.startwords = "startwords.json"
        with patch("builtins.open", unittest.mock.mock_open(read_data='{"hello": 0.5, "world": 0.5}')):
            sentence_prediction, word_prediction = self.predictor.predict(5, None)
            self.assertEqual(len(word_prediction.suggestions), 2)
            # self.assertEqual(word_prediction.suggestions[0].word, "hello")
            # self.assertEqual(word_prediction.suggestions[0].probability, 0.5)
            # self.assertEqual(word_prediction.suggestions[1].word, "world")
            # self.assertEqual(word_prediction.suggestions[1].probability, 0.5)

    def test_predict_with_tokens(self):
        self.context_tracker.get_tokens.return_value = 3
        self.predictor.db.ngram_like_table.return_value = [("test", 1)]
        self.predictor.db.unigram_counts_sum.return_value = 10
        self.predictor.db.ngram_count.side_effect = [1, 2, 3, 4, 5, 6]
        
        sentence_prediction, word_prediction = self.predictor.predict(5, None)
        
        # self.assertEqual(len(word_prediction.suggestions), 1)
        # self.assertEqual(word_prediction.suggestions[0].word, "test")
        # self.assertGreater(word_prediction.suggestions[0].probability, 0)

    def test_predict_exception_handling(self):
        self.context_tracker.get_tokens.return_value = 3
        self.predictor.db.ngram_like_table.side_effect = Exception("Test exception")
        
        sentence_prediction, word_prediction = self.predictor.predict(5, None)
        
        self.assertEqual(len(word_prediction.suggestions), 0)
        self.predictor.logger.debug.assert_called_with("Exception in SmoothedNgramPredictor predict function: Test exception")

if __name__ == "__main__":
    unittest.main()