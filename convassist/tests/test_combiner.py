# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from convassist.combiner.meritocrity_combiner import MeritocracyCombiner
from convassist.predictor.utilities.prediction import Prediction
from convassist.predictor.utilities.suggestion import Suggestion


class TestMeritocracyCombiner(unittest.TestCase):
    def setUp(self):
        self.combiner = MeritocracyCombiner()

    def _create_prediction(self, predictor_name="test_predictor"):
        prediction = Prediction()
        prediction.add_suggestion(Suggestion("Test", 0.3, predictor_name))
        prediction.add_suggestion(Suggestion("Test2", 0.3, predictor_name))
        prediction.add_suggestion(Suggestion("Test", 0.1, predictor_name))
        prediction.add_suggestion(Suggestion("Test3", 0.2, predictor_name))
        return prediction

    def _create_prediction2(self, predictor_name="test_predictor"):
        prediction = Prediction()
        prediction.add_suggestion(Suggestion("Test2", 0.3, predictor_name))
        prediction.add_suggestion(Suggestion("Test", 0.1, predictor_name))
        prediction.add_suggestion(Suggestion("Test3", 0.2, predictor_name))
        return prediction

    def test_filter(self):
        result = self.combiner.filter(self._create_prediction())

        correct = Prediction()
        correct.add_suggestion(Suggestion("Test3", 0.2, "test_predictor"))
        correct.add_suggestion(Suggestion("Test2", 0.3, "test_predictor"))
        correct.add_suggestion(Suggestion("Test", 0.4, "test_predictor"))

        assert result == correct

    def test_combine(self):
        predictions = [self._create_prediction2()]
        prediction2 = self._create_prediction2()
        prediction2.add_suggestion(Suggestion("Test4", 0.1, "test_predictor"))
        predictions.append(prediction2)
        _, result = self.combiner.combine(predictions, "")

        correct = Prediction()
        correct.add_suggestion(Suggestion("Test3", 0.4, "test_predictor"))
        correct.add_suggestion(Suggestion("Test2", 0.6, "test_predictor"))
        correct.add_suggestion(Suggestion("Test4", 0.1, "test_predictor"))
        correct.add_suggestion(Suggestion("Test", 0.2, "test_predictor"))

        assert result == correct

    def test_combine_with_sentence_prediction(self):
        predictions = [self._create_prediction2()]
        prediction2 = self._create_prediction2("SentenceCompletionPredictor")
        prediction2.add_suggestion(
            Suggestion("test is a sentence", 0.1, "SentenceCompletionPredictor")
        )
        predictions.append(prediction2)
        _, result = self.combiner.combine(predictions, "")

        correct = Prediction()
        correct.add_suggestion(Suggestion("Test2", 0.6, "test_predictor"))
        correct.add_suggestion(Suggestion("Test3", 0.4, "test_predictor"))
        correct.add_suggestion(Suggestion("Test", 0.2, "test_predictor"))
        correct.add_suggestion(
            Suggestion("test is a sentence", 0.1, "SentenceCompletionPredictor")
        )

        assert result == correct

    def test_computeLetterProbs_with_context(self):
        result = Prediction()
        result.add_suggestion(Suggestion("hello", 0.6, "TestPredictor"))
        result.add_suggestion(Suggestion("hi", 0.2, "TestPredictor"))
        context = "h"
        expected = [("e", 0.5), ("i", 0.5)]
        assert self.combiner.computeLetterProbs(result, context) == expected

    def test_computeLetterProbs_ignore_spell_correct_predictor(self):
        result = Prediction()
        result.add_suggestion(Suggestion("hello", 0.6, "SpellCorrectPredictor"))
        result.add_suggestion(Suggestion("hi", 0.2, "TestPredictor"))
        context = ""
        expected = [("h", 0.5)]
        assert self.combiner.computeLetterProbs(result, context) == expected

    def test_computeLetterProbs_ignore_sentence_completion_predictor(self):
        result = Prediction()
        result.add_suggestion(
            Suggestion("hello world", 0.6, predictor_name="SentenceCompletionPredictor")
        )
        result.add_suggestion(Suggestion("hi", 0.2, predictor_name="TestPredictor"))
        context = ""
        expected = [("h", 1.0)]
        assert self.combiner.computeLetterProbs(result, context) == expected
