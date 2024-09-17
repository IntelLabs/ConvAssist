# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from ConvAssist.combiner.meritocrity_combiner import MeritocracyCombiner
from ConvAssist.predictor.utilities.suggestion import Suggestion
from ConvAssist.predictor.utilities.prediction import Prediction


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
        prediction2.add_suggestion(Suggestion("test is a sentence", 0.1, "SentenceCompletionPredictor"))
        predictions.append(prediction2)
        _, result = self.combiner.combine(predictions, "")

        correct = Prediction()
        correct.add_suggestion(Suggestion("Test2", 0.6, "test_predictor"))
        correct.add_suggestion(Suggestion("Test3", 0.4, "test_predictor"))
        correct.add_suggestion(Suggestion("Test", 0.2, "test_predictor"))
        correct.add_suggestion(Suggestion("test is a sentence", 0.1, "SentenceCompletionPredictor"))

        assert result == correct
