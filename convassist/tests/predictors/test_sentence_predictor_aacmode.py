# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.sentence_completion_predictor import (
    SentenceCompletionPredictor,
)
from convassist.tests import setup_utils
from convassist.tests.predictors import TestPredictors
from convassist.predictor.utilities import PredictorResponse


class TestSentenceCompletionPredictor(TestPredictors):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()

    @classmethod
    def tearDownClass(cls):
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

        super().tearDownClass()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def setUp(self, mock_cuda, mock_mps):
        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()

        SOURCE_DIR = setup_utils.SOURCE_DIR

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
            "retrieveaac": "True",
            "sent_database": "sent_database.db",
            "retrieve_database": "all_aac.txt",
            "modelname": "IntelLabs/aac_gpt2",
            "tokenizer": "IntelLabs/aac_gpt2",
            "startsents": "startSentences.txt",
            "embedding_cache_path": "all_aac_embeddings.pkl",
            "sentence_transformer_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
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
        responses:PredictorResponse = self.predictor.predict(
            max_partial_prediction_size
        )
        self.assertIsNotNone(responses.sentencePredictions)
        self.assertLessEqual(len(responses.sentencePredictions), max_partial_prediction_size)
        self.assertIsNotNone(responses.wordPredictions)
        self.assertLessEqual(len(responses.sentencePredictions), max_partial_prediction_size)

    def test_learn_new_sentence(self):
        change_tokens = "This is a new sentence to learn"
        self.predictor.learn(change_tokens)

        self.predictor.context_tracker.context = "This is a new sentence"
        responses:PredictorResponse = self.predictor.predict(5)
        self.assertIsNotNone(responses.sentencePredictions)
        self.assertLessEqual(len(responses.sentencePredictions), 5)
        self.assertEqual(responses.sentencePredictions[0].word, " to learn") 


if __name__ == "__main__":
    unittest.main()
