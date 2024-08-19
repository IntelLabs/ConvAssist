# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from configparser import ConfigParser

from src.predictor.utilities.predictor_names import PredictorNames
from src.utilities.logging import ConvAssistLogger

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

    def __init__(self, config, logger=None):
        super().__init__()

        self.config: ConfigParser = config
        
        self._context_tracker = None
        self.set_predictors()
        if logger:
            self.logger = logger
        else:
            self.logger = ConvAssistLogger(__name__, level="DEBUG")

    @property
    def context_tracker(self):
        """The context_tracker property."""
        return self._context_tracker

    @context_tracker.setter
    def context_tracker(self, value):
        if self._context_tracker is not value:
            self._context_tracker = value
            self[:] = []
            self.set_predictors()

    @context_tracker.deleter
    def context_tracker(self):
        del self._context_tracker

    def set_predictors(self):
        if self.context_tracker:
            self[:] = []
            for predictor in self.config.get("PredictorRegistry", "predictors", fallback="").split():
                self.add_predictor(predictor)

    def add_predictor(self, predictor_name):
        from src.predictor.canned_phrases_predictor import CannedPhrasesPredictor
        from src.predictor.sentence_completion_predictor import SentenceCompletionPredictor
        from src.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
        from src.predictor.spell_correct_predictor import SpellCorrectPredictor

        predictor: Any = None
        
        if (self.config.get(predictor_name, "predictor_class") == "SmoothedNgramPredictor"
        ):
            predictor = SmoothedNgramPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                
                logger=self.logger
            )
        if (self.config.get(predictor_name, "predictor_class") == "SpellCorrectPredictor"
        ):
            predictor = SpellCorrectPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                
                logger=self.logger
            )

        if (self.config.get(predictor_name, "predictor_class") == "SentenceCompletionPredictor"
        ):
            predictor = SentenceCompletionPredictor(
                self.config, 
                self.context_tracker, 
                predictor_name, "gpt2",
                "gpt-2 model predictions", 
                logger=self.logger
            )


        if (self.config.get(predictor_name, "predictor_class") == "CannedPhrasesPredictor"
        ):
            predictor = CannedPhrasesPredictor(
                self.config, 
                self.context_tracker, 
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



    