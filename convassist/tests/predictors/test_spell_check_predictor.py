import configparser
import os
import unittest
from unittest.mock import patch

from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor.spell_correct_predictor import SpellCorrectPredictor

from .. import setup_utils
from . import TestPredictors


class TestSpellCorrectPredictor(TestPredictors):
    def setUp(self):
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
            "predictor_class": "SpellCorrectPredictor",
            "learn": "False",
        }

        self.context_tracker = ContextTracker(self.config)
        self.predictor = SpellCorrectPredictor(self.config, self.context_tracker, "test_predictor")

    def tearDown(self):
        setup_utils.teardown_static_resources()
        setup_utils.teardown_personalized_resources()

    def test_configure(self):
        self.assertIsNotNone(self.predictor)
        self.assertIsInstance(self.predictor, SpellCorrectPredictor)

    @parameterized.expand(
        [
            ("no_context", "", 0, None),
            ("spelled_correct", "hello", 1, "hello"),
            ("misspelled", "braekfast", 1, "breakfast"),
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

        if expected_word:
            self.assertEqual(word_predictions[0].word, expected_word)


if __name__ == "__main__":
    unittest.main()
