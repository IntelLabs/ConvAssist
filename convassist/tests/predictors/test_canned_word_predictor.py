# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.smoothed_ngram_predictor.canned_word_predictor import (
    CannedWordPredictor,
)

from convassist.tests import setup_utils
from convassist.tests.predictors import TestPredictors


class TestCannedWordPredictor(TestPredictors):
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
            "predictor_class": "CannedWordPredictor",
            "database": f"{SOURCE_DIR}/test_data/personalized/canned_ngram.db",
            # "sentences_db": f"{SOURCE_DIR}/test_data/personalized/canned_sentences.db",
            "learn": "True",
            "personalized_cannedphrases": "personalizedCannedPhrases.txt",
            "startwords": "startWords.json",
        }

        self.context_tracker = ContextTracker(self.config)
        self.predictor = CannedWordPredictor(self.config, self.context_tracker, "test_predictor")

    def tearDown(self):
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

    def test_configure(self):
        self.assertIsNotNone(self.predictor.svo_utils)
        self.assertIsNotNone(self.predictor.database)

    @parameterized.expand(
        [
            ("no_context", "", 1, "all"),
            ("trigram", "to the ", 1, "crazy"),
            ("trigram", "to the crazy", 1, "crazy"),
            ("bigram", "the ", 1, "crazy"),
            ("bigram", "the crazy", 1, "crazy"),
            ("unigram", "cra", 1, "crazy"),
            ("unigram", "crazy", 1, "crazy"),
        ]
    )
    def test_predict(self, name, context, max, expected_word):
        max_partial_prediction_size = max
        self.predictor.context_tracker.context = context
        sentence_predictions, word_predictions = self.predictor.predict(
            max_partial_prediction_size, None
        )
        self.assertIsNotNone(sentence_predictions)
        self.assertEqual(len(sentence_predictions), 0)
        self.assertIsNotNone(word_predictions)
        self.assertEqual(len(word_predictions), max_partial_prediction_size)
        self.assertEqual(word_predictions[0].word, expected_word)

    def test_learn_new_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)

        self.predictor.context_tracker.context = self.predictor.extract_svo("This is a new ")
        self.predictor.context_tracker.context = self.predictor.context_tracker.context + " "
        _, words = self.predictor.predict(1, None)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, "sentence")

    def test_learn_existing_sentence(self):
        change_tokens = "This is a new sentence to learn."
        self.predictor.learn(change_tokens)
        self.predictor.learn(change_tokens)

        self.predictor.context_tracker.context = self.predictor.extract_svo("This is a new ")
        self.predictor.context_tracker.context = self.predictor.context_tracker.context + " "
        _, words = self.predictor.predict(1, None)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, "sentence")

    # TODO: Implement this test
    # def test_remove_canned_words(self):

    #     with open(self.predictor.personalized_cannedphrases, "w") as f:
    #         f.write("My new phrase")

    #     self.predictor = None
    #     self.predictor = CannedWordPredictor(self.config, self.context_tracker, "test_predictor")

    #     self.predictor.context_tracker.context = "My new "
    #     _, words = self.predictor.predict(1, None)
    #     self.assertEqual(len(words), 1)
    #     self.assertEqual(words[0].word, "phrase")

    #     self.predictor.context_tracker.context = "to the "
    #     _, words = self.predictor.predict(1, None)
    #     self.assertEqual(len(words), 1)
    #     self.assertEqual(words[0].word, "crazy")


if __name__ == "__main__":
    unittest.main()
