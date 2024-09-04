# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from configparser import ConfigParser

# from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.predictor_names import PredictorNames
from ConvAssist.utilities.logging_utility import LoggingUtility

from ConvAssist.predictor.canned_phrases_predictor import CannedPhrasesPredictor
from ConvAssist.predictor.sentence_completion_predictor import SentenceCompletionPredictor
from ConvAssist.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from ConvAssist.predictor.spell_correct_predictor import SpellCorrectPredictor


class PredictorRegistry(list):
    """
    Manages instantiation and iteration through predictors and aids in
    generating predictions and learning.

    PredictorRegitry class holds the active predictors and provides the
    interface required to obtain an iterator to the predictors.

    The standard use case is: Predictor obtains an iterator from
    PredictorRegistry and invokes the predict() or learn() method on each
    Predictor pointed to by the iterator.

    Predictor registry should eventually just be a simple wrapper around
    plump.

    """

    def __init__(self, config, logger=None, context_tracker=None):
        super().__init__()

        self.config: ConfigParser = config
        self._context_tracker = None
        if logger:
            self.logger = logger
        else:
            self.logger = LoggingUtility.get_logger("predictor_registry", log_level=logging.DEBUG)

        self.set_predictors(context_tracker)

    # @property
    # def context_tracker(self) -> Any:
    #         """The context_tracker property."""
    #         return self._context_tracker

    # @context_tracker.setter
    # def context_tracker(self, value):
    #     if self._context_tracker is not value:
    #         self._context_tracker = value
    #         self[:] = []
    #         self.set_predictors()

    # @context_tracker.deleter
    # def context_tracker(self):
    #     del self._context_tracker

    def set_predictors(self, context_tracker=None):
        # if self.context_tracker:
        self[:] = []
        predictors = self.config.get("PredictorRegistry", "predictors", fallback="").split()
        for predictor in predictors:
            self.add_predictor(predictor, context_tracker)

    def add_predictor(self, predictor_name, context_tracker=None):
        predictor: Any = None
        
        if (self.config.get(predictor_name, "predictor_class") == "SmoothedNgramPredictor"
        ):
            predictor = SmoothedNgramPredictor(
                self.config,
                context_tracker,
                predictor_name,
                logger=self.logger
            )
        if (self.config.get(predictor_name, "predictor_class") == "SpellCorrectPredictor"
        ):
            predictor = SpellCorrectPredictor(
                self.config,
                context_tracker,
                predictor_name,
                logger=self.logger
            )

        if (self.config.get(predictor_name, "predictor_class") == "SentenceCompletionPredictor"
        ):
            predictor = SentenceCompletionPredictor(
                self.config, 
                context_tracker, 
                predictor_name, "gpt2",
                "gpt-2 model predictions", 
                logger=self.logger
            )


        if (self.config.get(predictor_name, "predictor_class") == "CannedPhrasesPredictor"
        ):
            predictor = CannedPhrasesPredictor(
                self.config, 
                context_tracker, 
                predictor_name, "gpt2",
                "gpt-2 model predictions", 
                logger=self.logger
            )

        if predictor:
            self.append(predictor)

    def model_status(self) -> int:
        model_status = 999
        for each in self:
            if(str(each).find(PredictorNames.SentenceComp.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    self.logger.info("SentenceCompletionPredictor model loaded")
                    model_status = 1
                else:
                    model_status = 0
            if(str(each).find(PredictorNames.CannedPhrases.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    model_status = 1
                    self.logger.info("CannedPhrasesPredictor model loaded")
                else:
                    model_status = 0
        return model_status

    def close_database(self):
        for predictor in self:
            predictor.close_database()



    