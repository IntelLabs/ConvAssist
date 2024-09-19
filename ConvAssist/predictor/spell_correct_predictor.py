# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from spellchecker import SpellChecker

from .predictor import Predictor
from .utilities import Prediction, Suggestion


class SpellCorrectPredictor(Predictor):
    def configure(self):
        pass

    def predict(self, max_partial_prediction_size=None, filter=None):
        token = self.context_tracker.get_last_token()
        setence_predictions = Prediction()
        word_predictions = Prediction()

        if token:
            spell = SpellChecker()
            suggestions = spell.candidates(token)
            if suggestions:
                for suggestion in suggestions:
                    prob = spell.word_usage_frequency(suggestion)
                    word_predictions.add_suggestion(
                        Suggestion(suggestion, prob, self.predictor_name)
                    )

        return setence_predictions, word_predictions[:max_partial_prediction_size]

    def learn_text(self, text):
        self.logger.warning("SpellCorrectPredictor does not support learning.")
