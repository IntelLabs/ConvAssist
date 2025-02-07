# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser

from convassist.predictor_registry import PredictorRegistry

from convassist.combiner.meritocrity_combiner import MeritocracyCombiner
from convassist.predictor.spell_correct_predictor import SpellCorrectPredictor
from convassist.predictor.utilities.prediction import UnknownCombinerException
from convassist.utilities.logging_utility import LoggingUtility


class PredictorActivator:
    """
    PredictorActivator starts the execution of the active predictors,
    monitors their execution and collects the predictions returned, or
    terminates a predictor's execution if it execedes its maximum
    prediction time.

    The predictions returned by the individual predictors are combined
    into a single prediction by the active Combiner.

    """

    def __init__(self, config, registry: PredictorRegistry, context_tracker=None, logger=None):
        self.config: ConfigParser = config
        self.registry = registry
        self.context_tracker = context_tracker
        self.combiner: MeritocracyCombiner
        self.max_partial_prediction_size = self.config.getint(
            "Selector", "suggestions", fallback=10
        )
        self.predict_time = None
        self._combination_policy = None

        if logger:
            self.logger = logger
        else:
            self.logger = LoggingUtility().get_logger(
                "predictor_activator", log_level=logging.DEBUG, queue_handler=True
            )

    @property
    def combination_policy(self):
        """The combination_policy property."""
        return self._combination_policy

    @combination_policy.setter
    def combination_policy(self, value):
        self._combination_policy = value
        if value.lower() == "meritocracy":
            self.combiner = MeritocracyCombiner()
        else:
            raise UnknownCombinerException()

    @combination_policy.deleter
    def combination_policy(self):
        del self._combination_policy

    def predict(self, multiplier=1, prediction_filter=None):
        sentence_predictions = []  # Store the predictions from the sentence predictor
        sentence_nextLetterProbs = (
            []
        )  # Store the combined next letter probabilities from the sentence predictor
        sentence_result = []  # Store the combined results from the sentence predictor

        word_predictions = []  # Store the predictions from the word predictor(s)
        word_nextLetterProbs = (
            []
        )  # Store the combined next letter probabilities from the word predictor(s)
        word_result = []  # Store the combined results from the word predictor(s)

        if not self.registry:
            self.logger.warning("No predictors registered.")
            return (word_nextLetterProbs, word_result, sentence_nextLetterProbs, sentence_result)

        if self.context_tracker:
            context = self.context_tracker.get_last_token()
        else:
            context = ""

        self.logger.info("Predicting next words and sentences")

        for predictor in self.registry:
            if type(predictor).__name__ == SpellCorrectPredictor.__name__:
                continue
            try:
                self.logger.info(
                    f"Predictor {predictor.predictor_name} - Predicting next {self.max_partial_prediction_size} words and sentences"
                )
                # Get sentences and/or words from the predictor
                sentences, words = predictor.predict(
                    self.max_partial_prediction_size * multiplier, prediction_filter
                )

                # Append the sentences to the sentence_predictions list
                if sentences:
                    sentence_predictions.append(sentences)

                # Append the words to the word_predictions list
                if words:
                    word_predictions.append(words)

                self.logger.info(
                    f"Predictor {predictor.predictor_name} - Predicted {len(sentences)} sentences and {len(words)} words"
                )

            except Exception as e:
                self.logger.critical(f"Predictor {predictor.predictor_name}: {e}", exc_info=True, stack_info=True)
                continue

        # If the word predictor(s) return empty lists, use predictions from the spell predictor
        if word_predictions == []:

            spellingPredictor = self.registry.get_predictor(SpellCorrectPredictor.__name__)
            if spellingPredictor:
                _, words = spellingPredictor.predict(
                    self.max_partial_prediction_size * multiplier, prediction_filter
                )

                if words:
                    word_predictions.append(words)

        # Combine the sentence predictions and get the next sentence letter probabilities
        sentence_nextLetterProbs, sentence_result = self.combiner.combine(
            sentence_predictions, context
        )

        # Combine the word predictions and get the next word letter probabilities
        word_nextLetterProbs, word_result = self.combiner.combine(word_predictions, context)

        self.logger.info(
            f"Predictions completed. Returning {len(word_result)} words and {len(sentence_result)} sentences."
        )
        return (word_nextLetterProbs, word_result, sentence_nextLetterProbs, sentence_result)

    def recreate_database(self):  # pragma: no cover
        for predictor in self.registry:
            predictor.recreate_database()

    def update_params(self, test_gen_sentence_pred, retrieve_from_AAC):  # pragma: no cover
        for predictor in self.registry:
            predictor.load_model()

    def read_updated_toxicWords(self):  # pragma: no cover
        for predictor in self.registry:
            predictor.read_personalized_toxic_words()

    def learn_text(self, text):  # pragma: no cover
        for predictor in self.registry:
            predictor.learn(text)
