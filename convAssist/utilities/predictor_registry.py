# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ConvAssist.predictor import canned_phrases_predictor, \
                    sentence_completion_predictor, \
                    smoothed_ngram_predictor, \
                    spell_correct_predictor
from ConvAssist.predictor.utilities.predictor_names import PredictorNames
from ConvAssist.utilities.logging import ConvAssistLogger

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

    def __init__(self, config, dbconnection=None):
        self.config = config
        self.dbconnection = dbconnection
        self._context_tracker = None
        self.set_predictors()
        self.log = ConvAssistLogger(__name__, level=ConvAssistLogger.DEBUG)

    def context_tracker():
        doc = "The context_tracker property."

        def fget(self):
            return self._context_tracker

        def fset(self, value):
            if self._context_tracker is not value:
                self._context_tracker = value
                self[:] = []
                self.set_predictors()

        def fdel(self):
            del self._context_tracker

        return locals()

    context_tracker = property(**context_tracker())

    def set_predictors(self):
        if self.context_tracker:
            self[:] = []
            for predictor in self.config.get("PredictorRegistry", "predictors").split():
                self.add_predictor(predictor)

    def add_predictor(self, predictor_name):
        predictor = None
        
        if (self.config.get(predictor_name, "predictor_class") == "SmoothedNgramPredictor"
        ):
            predictor = smoothed_ngram_predictor.SmoothedNgramPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                dbconnection=self.dbconnection,
            )
        if (self.config.get(predictor_name, "predictor_class") == "SpellCorrectPredictor"
        ):
            predictor = spell_correct_predictor.SpellCorrectPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                dbconnection=self.dbconnection,
            )

        if (self.config.get(predictor_name, "predictor_class") == "SentenceCompletionPredictor"
        ):
            predictor = sentence_completion_predictor.SentenceCompletionPredictor(
                self.config, self.context_tracker, predictor_name, "gpt2",
                "gpt-2 model predictions", self.dbconnection)


        if (self.config.get(predictor_name, "predictor_class") == "CannedPhrasesPredictor"
        ):
            predictor = canned_phrases_predictor.CannedPhrasesPredictor(
                self.config, self.context_tracker, predictor_name, "gpt2",
                                           "gpt-2 model predictions", self.dbconnection)

        if predictor:
            self.append(predictor)

    def model_status(self):
        model_status = 999
        for each in self:
            if(str(each).find(PredictorNames.SentenceComp.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    self.log.info("SentenceCompletionPredictor model loaded")
                    model_status = 1
                else:
                    model_status = 0
            if(str(each).find(PredictorNames.CannedPhrases.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    model_status = 1
                    self.log.info("CannedPhrasesPredictor model loaded")
                else:
                    model_status = 0
        return model_status

    def close_database(self):
        for predictor in self:
            predictor.close_database()



    