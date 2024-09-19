# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from spellchecker import SpellChecker

import ConvAssist.predictor.utilities.prediction as prediction
from ConvAssist.predictor.predictor import Predictor


class SpellCorrectPredictor(Predictor):
    def configure(self):
        pass

    def predict(self, max_partial_prediction_size=None, filter=None):
        token = self.context_tracker.get_last_token()
        setence_predictions = prediction.Prediction()
        word_predictions = prediction.Prediction()

        if token:
            spell = SpellChecker()
            suggestions = spell.candidates(token)
            if suggestions:
                for suggestion in suggestions:
                    prob = spell.word_usage_frequency(suggestion)
                    word_predictions.add_suggestion(
                        prediction.Suggestion(suggestion, prob, self.predictor_name)
                    )

        return setence_predictions, word_predictions[:max_partial_prediction_size]

    def learn_text(self, text):
        self.logger.warning("SpellCorrectPredictor does not support learning.")
