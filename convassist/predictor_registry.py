# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser
from typing import Any

from convassist.context_tracker import ContextTracker
from convassist.predictor.canned_phrases_predictor import CannedPhrasesPredictor
from convassist.predictor.sentence_completion_predictor import (
    SentenceCompletionPredictor,
)
from convassist.predictor.spell_correct_predictor import SpellCorrectPredictor
# from convassist.predictor.keyword_generator_predictor import KeywordGeneratorPredictor
# from convassist.predictor.response_generator_predictor import KeywordResponseGeneratorPredictor
from convassist.predictor.contextaware_predictor import ContextAwarePredictor
from convassist.predictor.smoothed_ngram_predictor.canned_word_predictor import CannedWordPredictor
from convassist.predictor.smoothed_ngram_predictor.general_word_predictor import GeneralWordPredictor
from convassist.predictor.smoothed_ngram_predictor.smoothed_ngram_predictor import (
    SmoothedNgramPredictor,
)

predictors = {
    "ShortHandPredictor": SmoothedNgramPredictor,
    "SmoothedNgramPredictor": SmoothedNgramPredictor,
    "CannedWordPredictor": CannedWordPredictor,
    "GeneralWordPredictor": GeneralWordPredictor,
    "SpellCorrectPredictor": SpellCorrectPredictor,
    "SentenceCompletionPredictor": SentenceCompletionPredictor,
    "CannedPhrasesPredictor": CannedPhrasesPredictor,
    "CRGPredictor": ContextAwarePredictor,
    # "KeywordGeneratorPredictor": KeywordGeneratorPredictor,
    # "KeywordResponseGeneratorPredictor": KeywordResponseGeneratorPredictor,
}


class PredictorRegistry(list):
    """
    Manages instantiation and iteration through predictors and aids in
    generating predictions and learning.

    PredictorRegitry class holds the active predictors and provides the
    interface required to obtain an iterator to the predictors.

    The standard use case is: Predictor obtains an iterator from
    PredictorRegistry and invokes the predict() or learn() method on each
    Predictor pointed to by the iterator.
    """

    def __init__(self):
        super().__init__()

    def set_predictors(
        self,
        config: ConfigParser,
        context_tracker: ContextTracker,
        logger: logging.Logger,
        predictors: list[str] | None = None,
    ):
        self[:] = []

        if not predictors:
            predictors = config.get("PredictorRegistry", "predictors", fallback="").split()

        for predictor in predictors:
            print("predictor", predictor)
            self._add_predictor(predictor, config, context_tracker, logger)

    def _add_predictor(
        self,
        predictor_name,
        config: ConfigParser,
        context_tracker: ContextTracker,
        logger: logging.Logger,
    ):
        predictor: Any = None

        predictor_class = self.get_predictor_class(predictor_name, config)
        print("predictor_class", predictor_class)
        if predictor_class in predictors:
            try:
                if predictor_class in predictors:
                    predictor = predictors[predictor_class](
                        config, context_tracker, predictor_name, logger
                    )
                else:
                    logger.error(
                        f"Predictor class {predictor_class} is not found in the predictor_mapping dictionary."
                    )

            except TypeError as e:
                logger.error(f"Error instantiating predictor {predictor_name}: {e}")
                return
        if predictor:
            self.append(predictor)

    def get_predictor_class(self, predictor_name, config):

        #TODO: Fix this hack. 
        # This is a hack to get the predictor class from the config file.
        # The config file should have a mapping of predictor_name to predictor_class
        # but two predictor classes were renamed and the config file was not updated.
        # This hack will be removed once the config file is updated.

        if predictor_name == "CannedWordPredictor":
            return "CannedWordPredictor"
        elif predictor_name == "DefaultSmoothedNgramPredictor":
            return "GeneralWordPredictor"
        else: 
            return config.get(predictor_name, "predictor_class")

    def model_status(self) -> bool:
        model_status = False

        sentPredictor = self.get_predictor("SentenceCompletionPredictor")
        if type(sentPredictor) is SentenceCompletionPredictor:
            model_status = model_status and sentPredictor.model_loaded

        cannedPredictor = self.get_predictor("CannedPhrasesPredictor")
        if type(cannedPredictor) is CannedPhrasesPredictor:
            model_status = model_status and cannedPredictor.model_loaded

        return model_status

    def get_predictor(self, predictor_name):
        for predictor in self:
            if predictor.predictor_name == predictor_name:
                return predictor
        return None

    def list_predictors(self):
        result = []
        for predictor in predictors:
            if self.get_predictor(predictor):
                result.append(f"{predictor}")

        return result
