# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ConvAssist.utilities.logging import ConvAssistLogger
from ConvAssist.utilities.singleton import PredictorSingleton

__all__ = ["Predictor", 
           "SmothedNgramPredictor", 
           "SpllCorrectPredictor", 
           "WordPredictor", 
           "SentencePredictor",
           "Prediction"]


class Predictor(metaclass=PredictorSingleton):
    """
    Base class for predictors.

    """
    # singleton = None
    # def __new__(cls):
    #     if not cls.singleton:
    #         cls.singleton = super().__new__(cls)
    #     return cls.singleton

    def __init__(
            self, 
            config, 
            context_tracker, 
            predictor_name, 
            short_desc=None, 
            long_desc=None,
            dbconnection=None,
            logger=None
    ):
        self.config = config
        self.context_tracker = context_tracker
        self.predictor_name = predictor_name
        self.short_description = short_desc
        self.long_description = long_desc
        self.dbconnection = dbconnection
        
        #configure a logger
        if logger:
            self.log = logger
        else:
            self.log = ConvAssistLogger(name=self.predictor_name, 
                                        level=ConvAssistLogger.DEBUG)

    def token_satifies_filter(token, prefix, token_filter):
        if token_filter:
            for char in token_filter:
                candidate = prefix + char
                if token.startswith(candidate):
                    return True
        return False
    
    def get_name(self):
        return self.predictor_name
    
    def get_description(self):
        return self.long_description
    
    def get_long_description(self):
        return self.long_description
    
    def predict(self, max_partial_prediction_size = None, filter = None):
        raise NotImplementedError("Subclasses must implement this method")
    
    def learn(self, change_tokens = None):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _read_config(self):
        raise NotImplementedError("Subclasses must implement this method")