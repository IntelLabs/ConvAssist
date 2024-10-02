# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from spellchecker import SpellChecker

from .predictor import Predictor
from .utilities import Prediction, Suggestion


class SpellCorrectPredictor(Predictor):
    """
    SpellCorrectPredictor is a class that extends the Predictor class to provide
    spell correction functionality. It uses a spell checker to generate suggestions
    for the last token in the context.
    Methods:
        configure():
            Placeholder method for configuration. Currently does nothing.
        predict(max_partial_prediction_size=None, filter=None):
            Generates spell correction suggestions for the last token in the context.
            Args:
                max_partial_prediction_size (int, optional): Maximum number of suggestions to return.
                filter (optional): Not used in this implementation.
            Returns:
                tuple: A tuple containing setence_predictions and word_predictions.
                       setence_predictions is currently an empty Prediction object.
                       word_predictions is a Prediction object containing suggestions
                       for the last token, limited by max_partial_prediction_size.
        learn_text(text):
            Logs a warning that SpellCorrectPredictor does not support learning.
            Args:
                text (str): The text to learn from. Not used in this implementation.
    """

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
