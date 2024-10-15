import configparser
import os
import time
import unittest
from unittest.mock import patch

from convassist.context_tracker import ContextTracker
from convassist.predictor.canned_phrases_predictor import CannedPhrasesPredictor

from .. import setup_utils
from . import TestPredictors


class TestCannedPhrasesPredictor(TestPredictors):
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def setUp(self, mock_cuda, mock_mps):
        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()

        self.config = configparser.ConfigParser()
        self.config["Common"] = {
            "static_resources_path": f"{setup_utils.SOURCE_DIR}/test_data/static",
            "personalized_resources_path": f"{setup_utils.SOURCE_DIR}/test_data/personalized",
            "deltas": "0.01 0.1 0.89",
            "stopwords": "stopwords.txt",
        }
        self.config["PredictorRegistry"] = {
            "predictors": "test_predictor",
        }
        self.config["test_predictor"] = {
            "predictor_class": "CannedPhrasesPredictor",
            "generic_phrases": "cannedPhrases.txt",
            "sentences_db": "canned_sentences.db",
            "personalized_cannedphrases": "personalizedCannedPhrases.txt",
            "learn": "True",
            "embedding_cache_path": "personalizedCannedPhrases_embeddings.pkl",
            "index_path": "hnswlib_canned.index",
            "sbertmodel": "multi-qa-MiniLM-L6-cos-v1",
        }

        self.config["ContextTracker"] = {"lowercase_mode": "True"}

        self.context_tracker = ContextTracker(self.config)
        self.predictor = CannedPhrasesPredictor(
            self.config, self.context_tracker, "test_predictor"
        )

    def tearDown(self):
        time.sleep(3)
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

    def test_configure(self):
        self.predictor.configure()

        self.assertTrue(self.predictor._model_loaded)
        self.assertEqual(self.predictor.device, "cpu")
        self.assertEqual(self.predictor.n_gpu, 0)

    def test_load_model(self):

        self.predictor.load_model()
        self.assertTrue(self.predictor._model_loaded)

    def test_predict_no_context(self):
        max_partial_prediction_size = 1
        self.predictor.load_model()
        self.predictor.context_tracker.context = ""
        sentence_predictions, word_predictions = self.predictor.predict(
            max_partial_prediction_size
        )
        self.assertIsNotNone(sentence_predictions)
        self.assertEqual(len(sentence_predictions), max_partial_prediction_size)
        self.assertIsNotNone(word_predictions)
        self.assertEqual(len(word_predictions), 0)

    def test_predict_context(self):
        max_partial_prediction_size = 1
        self.predictor.load_model()
        self.predictor.context_tracker.context = "Here's to the "

        sentence_predictions, word_predictions = self.predictor.predict(
            max_partial_prediction_size
        )

        self.assertIsNotNone(sentence_predictions)
        self.assertEqual(len(sentence_predictions), max_partial_prediction_size)
        self.assertIsNotNone(word_predictions)
        self.assertEqual(len(word_predictions), 0)

        self.assertEqual(sentence_predictions[0].word, "Here's to the crazy ones")

    def test_learn_new_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)

        self.predictor.context_tracker.context = "This is a new "
        sentences, _ = self.predictor.predict(1)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].word, "This is a new sentence to learn.")

    def test_learn_existing_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)
        self.predictor.learn(change_tokens)

        self.predictor.context_tracker.context = "This is a new "
        sentences, _ = self.predictor.predict(1)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].word, "This is a new sentence to learn.")


if __name__ == "__main__":
    unittest.main()
