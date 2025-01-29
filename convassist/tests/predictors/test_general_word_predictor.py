# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import os
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.smoothed_ngram_predictor.general_word_predictor import (
    GeneralWordPredictor,
)
from convassist.utilities.ngram.ngramutil import NGramUtil

from convassist import setup_utils
from convassist import TestPredictors


class TestGeneralWordPredictor(TestPredictors):
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def setUp(self, mock_cuda, mock_mps):
        setup_utils.setup_static_resources()
        setup_utils.setup_personalized_resources()
        setup_utils.setup_test_words_db()

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
            "predictor_class": "GeneralWordPredictor",
            "database": "test_words.db",
            "learn": "False",
            "aac_dataset": "all_aac.txt",
            "startwords": "startWords.json",
        }

        self.context_tracker = ContextTracker(self.config)
        self.predictor = GeneralWordPredictor(self.config, self.context_tracker, "test_predictor")

    def tearDown(self):
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

    def test_configure(self):
        self.assertIsNotNone(self.predictor.aac_dataset)
        self.assertIsNotNone(self.predictor.database)


    @parameterized.expand(
        [
            ("no_context", "", 1, "all"),
            ("3-gram_whole_word", "in the ", 1, "square"),
            ("2-gram_whole_word", "the ", 1, "crazy"),
            ("3-gram_partial_word", "in the sq", 1, "square"),
            ("2-gram_partial_word", "the sq", 1, "square"),
            ("1-gram_partial_word", "sq", 1, "square"),
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
        # self.assertEqual(len(word_predictions), max_partial_prediction_size)
        self.assertEqual(word_predictions[0].word, expected_word)


if __name__ == "__main__":
    unittest.main()
