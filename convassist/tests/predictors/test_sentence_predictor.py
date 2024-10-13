import configparser
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.sentence_completion_predictor import (
    SentenceCompletionPredictor,
)

from .. import setup_utils


class TestSentenceCompletionPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass: This runs once before any tests in the class.")
        cls.shared_resource = "Shared Resource"
        SOURCE_DIR = setup_utils.SOURCE_DIR

        cls.config = configparser.ConfigParser()
        cls.config["Common"] = {
            "static_resources_path": f"{SOURCE_DIR}/test_data/static",
            "personalized_resources_path": f"{SOURCE_DIR}/test_data/personalized",
            "deltas": "0.01 0.1 0.89",
            "stopwords": "stopwords.txt",
        }
        cls.config["PredictorRegistry"] = {
            "predictors": "test_predictor",
        }
        cls.config["test_predictor"] = {
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
        cls.config["ContextTracker"] = {"lowercase_mode": "True"}

        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass: This runs once after all tests in the class.")
        del cls.shared_resource
        del cls.config

        setup_utils.teardown_personalized_resources()
        setup_utils.teardown_static_resources()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def setUp(self, mock_cuda, mock_mps):

        self.context_tracker = ContextTracker(self.config)
        self.predictor = SentenceCompletionPredictor(
            self.config, self.context_tracker, "test_predictor"
        )

    def test_configure(self):
        self.assertTrue(self.predictor._model_loaded)
        self.assertEqual(self.predictor.device, "cpu")
        self.assertEqual(self.predictor.n_gpu, 0)

    def test_load_model(self):
        self.predictor.load_model()
        self.assertTrue(self.predictor.model_loaded)

    @parameterized.expand(
        [
            ("no_context", "", 5),
            ("context_with_space", "i am thirsty ", 5),
            ("context_with_partial_word", "i am thir", 5),
        ]
    )
    def test_predict(self, name, context, max):
        max_partial_prediction_size = max
        self.predictor.context_tracker.context = context
        sentence_predictions, word_predictions = self.predictor.predict(
            max_partial_prediction_size
        )
        self.assertIsNotNone(sentence_predictions)
        self.assertLessEqual(len(sentence_predictions), max_partial_prediction_size)
        self.assertIsNotNone(word_predictions)
        self.assertLessEqual(len(sentence_predictions), max_partial_prediction_size)

    def test_learn_new_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)

        sentences = self.predictor._retrieve_fromDataset(change_tokens)
        self.assertEqual(len(sentences), 1)

    def test_learn_existing_sentence(self):
        change_tokens = "This is an existing sentence."
        self.predictor.learn(change_tokens)
        self.predictor.learn(change_tokens)

        sentences = self.predictor._retrieve_fromDataset(change_tokens)
        self.assertEqual(len(sentences), 1)


if __name__ == "__main__":
    unittest.main()
