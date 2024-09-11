# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from configparser import ConfigParser

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.predictor_names import PredictorNames

from ConvAssist.predictor.canned_phrases_predictor import CannedPhrasesPredictor
from ConvAssist.predictor.sentence_completion_predictor import SentenceCompletionPredictor
from ConvAssist.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from ConvAssist.predictor.spell_correct_predictor import SpellCorrectPredictor

predictor_mapping = {
    "SmoothedNgramPredictor": SmoothedNgramPredictor,
    "SpellCorrectPredictor": SpellCorrectPredictor,
    "SentenceCompletionPredictor": SentenceCompletionPredictor,
    "CannedPhrasesPredictor": CannedPhrasesPredictor
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

    def set_predictors(self, config: ConfigParser, context_tracker: ContextTracker, logger: logging.Logger):
        self[:] = []

        predictors = config.get("PredictorRegistry", "predictors", fallback="").split()

        for predictor in predictors:
            self._add_predictor(predictor, config, context_tracker, logger)

    def _add_predictor(self, predictor_name, config: ConfigParser, context_tracker: ContextTracker, logger: logging.Logger):
        predictor: Any = None
        
        predictor_class = config.get(predictor_name, "predictor_class")

        if predictor_class in predictor_mapping:
            try:
                if predictor_class in predictor_mapping:
                    predictor = predictor_mapping[predictor_class](config, context_tracker, predictor_name, logger)
                else:
                    logger.error(f"Predictor class {predictor_class} is not found in the predictor_mapping dictionary.")

            except TypeError as e:
                logger.error(f"Error instantiating predictor {predictor_name}: {e}")
                return
        if predictor:
            self.append(predictor)

    def model_status(self) -> int:
        model_status = 999
        for each in self:
            if(str(each).find(PredictorNames.SentenceComp.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    model_status = 1
                else:
                    model_status = 0
            if(str(each).find(PredictorNames.CannedPhrases.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    model_status = 1
                else:
                    model_status = 0
        return model_status

    def get_predictor(self, predictor_name):
        for predictor in self:
            if predictor.predictor_name == predictor_name:
                return predictor
        return None

    