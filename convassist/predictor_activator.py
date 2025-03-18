# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser

from convassist.predictor_registry import PredictorRegistry

from convassist.predictor.spell_correct_predictor import SpellCorrectPredictor
from convassist.predictor.utilities import PredictorResponse
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

    def predict(self, multiplier=1, prediction_filter=None) ->  PredictorResponse:
        combined_responses = PredictorResponse() # Store the combined results from the predictors

        if not self.registry:
            self.logger.warning("No predictors registered.")
            # We know it's empty, so return an empty response without calling combine.
            return combined_responses

        if self.context_tracker:
            context = self.context_tracker.get_last_token()
        else:
            context = ""

        self.logger.info("Predicting next words, sentences and/or keywords")

        for predictor in self.registry:
            if type(predictor).__name__ == SpellCorrectPredictor.__name__:
                continue
            try:
                self.logger.info(
                    f"Predictor {predictor.predictor_name} - Predicting next {self.max_partial_prediction_size}"
                )
                combined_responses.extend(predictor.predict(
                    self.max_partial_prediction_size * multiplier, prediction_filter
                ))

            except TypeError as e:
                self.logger.critical(f"Predictor {predictor.predictor_name} not implemented correctly: {e}", exc_info=True, stack_info=True)
                continue

            except Exception as e:
                self.logger.critical(f"Predictor {predictor.predictor_name}: {e}", exc_info=True, stack_info=True)
                continue

        # # If the word predictor(s) return empty lists, use predictions from the spell predictor
        # if combined_responses.wordPredictions == []:

        #     spellingPredictor = self.registry.get_predictor(SpellCorrectPredictor.__name__)
        #     if spellingPredictor and spellingPredictor.predictor_name == SpellCorrectPredictor.__name__:
        #         _, words = spellingPredictor.predict(
        #             self.max_partial_prediction_size * multiplier, prediction_filter
        #         )

        #         if words:
        #             combined_responses.wordPredictions.append(words)

        return combined_responses.combine(context)

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
    

