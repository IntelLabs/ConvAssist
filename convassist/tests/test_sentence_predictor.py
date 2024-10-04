import configparser
import os
import unittest
from unittest.mock import patch

from convassist.context_tracker import ContextTracker
from convassist.predictor.sentence_completion_predictor import (
    SentenceCompletionPredictor,
)

from . import setup_utils

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSentenceCompletionPredictor(unittest.TestCase):
    def setUp(self):
        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()

        self.config = configparser.ConfigParser()
        self.config["Common"] = {
            "static_resources_path": f"{SOURCE_DIR}/test_data/static",
            "personalized_resources_path": f"{SOURCE_DIR}/test_data/personalized",
            "deltas": "0.01 0.1 0.89",
            "stopwords": "stopwords.txt",
        }
        self.config["PredictorRegistry"] = {
            "predictors": "test_predictor",
        }
        self.config["test_predictor"] = {
            "predictor_class": "SentenceCompletionPredictor",
            "learn": "True",
            "test_generalsentenceprediction": "False",
            "retrieveaac": "False",
            "sent_database": "sent_database.db",
            "retrieve_database": "all_aac.txt",
            "modelname": "aac_gpt2",
            "tokenizer": "aac_gpt2",
            "startsents": "startSentences.txt",
            "embedding_cache_path": "all_aac_embeddings.pkl",
            "sentence_transformer_model": "multi-qa-MiniLM-L6-cos-v1",
            "index_path": "all_aac_semanticSearch.index",
            "blacklist_file": "filter_words.txt",
            "stopwords": "stopwords.txt",
            "personalized_allowed_toxicwords_file": "personalized_allowed_toxicwords.txt",
        }
        self.config["ContextTracker"] = {"lowercase_mode": "True"}

        self.context_tracker = ContextTracker(self.config)
        self.predictor = SentenceCompletionPredictor(
            self.config, self.context_tracker, "test_predictor"
        )

    def tearDown(self):
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_configure(self, mock_cuda, mock_mps):
        self.predictor.configure()

        self.assertTrue(self.predictor._model_loaded)
        self.assertEqual(self.predictor.device, "cpu")
        self.assertEqual(self.predictor.n_gpu, 0)

    def test_load_model(self):

        self.predictor.load_model()
        self.assertTrue(self.predictor._model_loaded)

    def test_generate(self):
        context = "I am thirsty. can you"
        num_gen = 1

        if not self.predictor._model_loaded:
            self.predictor.load_model()

        predictions = self.predictor._generate(context, num_gen)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(
            repr(predictions[0]), "Suggestion:  help me? - Probability: 0.28700759013493854"
        )

    def test_predict(self):
        max_partial_prediction_size = 5
        self.predictor.load_model()
        self.predictor.context_tracker.context = "i am thirsty. can you "
        sentence_predictions, word_predictions = self.predictor.predict(
            max_partial_prediction_size
        )
        self.assertIsNotNone(sentence_predictions)
        self.assertIsNotNone(word_predictions)

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
        self.assertEqual(len(sentence_predictions), max_partial_prediction_size)

    def test_learn_new_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)

        sentences = self.predictor._retrieve_fromDataset(change_tokens)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].probability, 1.0)

    def test_learn_existing_sentence(self):
        change_tokens = "This is an existing sentence."
        self.predictor.learn(change_tokens)
        self.predictor.learn(change_tokens)

        sentences = self.predictor._retrieve_fromDataset(change_tokens)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].probability, 1.0)


if __name__ == "__main__":
    unittest.main()
