# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from configparser import ConfigParser
from typing import Any
from nltk import sent_tokenize
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor_registry import PredictorRegistry
from ConvAssist.predictior_activator import PredictorActivator
from ConvAssist.utilities.callback import BufferedCallback
import logging
from ConvAssist.utilities.logging_utility import LoggingUtility

class ConvAssist:
    """
    ConvAssist class represents a conversational assistant.
    """
    def __init__(self, id: str, ini_file: str,
                 config:ConfigParser | None = None, 
                 callback:BufferedCallback | None = None,  
                 log_location:str | None = None, 
                 log_level:int | None = None):
 
        self.config = config
        self.callback = callback
        self.log_level = log_level 
        self.log_location = log_location
        self.initialized = False
        self.id = id
        self.ini_file = ini_file

        if self.config and self.callback:
            self.initialize(self.config, self.callback, self.log_location, self.log_level)

    def initialize(self, config:ConfigParser, callback:BufferedCallback, log_location:str|None = None, log_level:int|None = None):
        if not config:
            raise AttributeError("Config not provided.")
        
        if not self.config:
            self.config = config
            
        if log_level:
            self.log_level = log_level
        else:
            self.log_level = logging.getLevelName(self.config.get("Logging", "log_level", fallback="DEBUG")) # default to info level logging

        if log_location:
            self.log_location = log_location
        else:
            self.log_location = self.config.get("Logging", "log_location", fallback="") # default to no log file

        # TODO: MAKE PASSING LOGGER IN THE INITIALIZATION OPTIONAL
        self.logger = LoggingUtility().get_logger(self.id, self.log_level, self.log_location, True)

            
        lowercase_mode = self.config.getboolean("ContextTracker", "lowercase_mode", fallback=False)
        self.context_tracker = ContextTracker(lowercase_mode, callback)

        self.predictor_registry = PredictorRegistry(
                self.config, self.logger, self.context_tracker
            )
    

        self.predictor_activator = PredictorActivator(
                self.config, self.predictor_registry, self.context_tracker
            )
        self.predictor_activator.combination_policy = "meritocracy"

        self.initialized = True

    # Predict function - calls the predict on predictor_activator (this calls
    # language model predictions on each of the defined predictor)

    def predict(self):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")

        multiplier = 1
        (wordprob, word, sentprob, sent) = self.predictor_activator.predict(multiplier)
        if word!=[]:
            #### normalize word probabilites over 10 words. 
            normalized_words = word[0:10]
            prob_sum_over10 = 0.0
            words = []
            for w in enumerate(normalized_words):
                prob_sum_over10 += w[1].probability
                words.append(w[1].word)

        return (wordprob, [(p.word, p.probability/prob_sum_over10) for p in word], sentprob, [(p.word, p.probability) for p in sent])

    # Paramters updated in ACAT are synced with the parameters initialized 
    # in ConvAssist Predictors.    
    def update_params(self, test_gen_sentence_pred, retrieve_from_AAC):
        self.predictor_activator.update_params(test_gen_sentence_pred, retrieve_from_AAC)


    def read_updated_toxicWords(self):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")
        
        self.predictor_activator.read_updated_toxicWords()

    """
    This function is called from ACAT --> NamedPipeApplication --> ConvAssist to set the Logs location for ConvAssist's logger
    """
    def setLogLocation(self, filename, pathLoc , level):
        # self.predictor_activator.set_log(filename, pathLoc, level)
        pass

    """
    This function is called from NamedPipeApplication --> ConvAssist to Recreate Databases - for the cannedPhrases predictor. 
    """
    def cannedPhrase_recreateDB(self):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")
            
        self.predictor_activator.recreate_canned_phrasesDB()

    """
    This function is called from ACAT --> NamedPipeApplication --> ConvAssist to "learn" a sentence, word or phrase
    """
    def learn_db(self, text):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")
        
        sentences = sent_tokenize(text)
        for eachSent in sentences:
            self.predictor_activator.learn_text(eachSent)

    """
    This function is called from NamedPipeApplication --> ConvAssist 
    Checks if models associated with a predictor is loaded
    """
    def check_model(self):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")
        
        status = self.predictor_registry.model_status()
        return status

    def close_database(self):
        if not self.initialized:
            raise AttributeError("ConvAssist {self.name} not initialized.")
        
        self.predictor_registry.close_database()