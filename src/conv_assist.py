# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from configparser import ConfigParser
from typing import Any
from nltk import sent_tokenize
from src.context_tracker import ContextTracker
from src.predictor_registry import PredictorRegistry
from src.predictor.utilities.predictior_activator import PredictorActivator
from src.utilities.logging import ConvAssistLogger

class ConvAssist:
    """
    ConvAssist class represents a conversational assistant.

    Attributes:
        config (object): The configuration object for ConvAssist.
        callback (object): The callback object for ConvAssist.
        predictor_registry (object): The predictor registry for ConvAssist.
        context_tracker (object): The context tracker for ConvAssist.
        predictor_activator (object): The predictor activator for ConvAssist.

    Methods:
        __init__(self, callback, config): Initializes a new instance of the ConvAssist class.
        predict(self): Calls the predict function on the predictor_activator.
        update_params(self, test_gen_sentence_pred, retrieve_from_AAC): Updates the parameters in ACAT and syncs them with the initialized parameters in ConvAssist predictors.
        read_updated_toxicWords(self): Reads the updated toxic words from the predictor_activator.
        setLogLocation(self, filename, pathLoc, level): Sets the log location for ConvAssist's logger.
        cannedPhrase_recreateDB(self): Recreates the databases for the cannedPhrases predictor.
        learn_db(self, text): Learns a sentence, word, or phrase.
        check_model(self): Checks if the models associated with a predictor are loaded.
        close_database(self): Closes the database for ConvAssist.
    """
    def __init__(self, callback, config:ConfigParser):
        self.config = config
        self.callback = callback
        self.log_location = self.config.get("Logging", "log_location") # default to no log file
        self.log_level = self.config.get("Logging", "log_level") # default to info level logging
        self.logger = ConvAssistLogger(self.log_location, self.log_level)
        
        self.predictor_registry = PredictorRegistry(
            self.config, self.logger
        )
        self.context_tracker = ContextTracker(
            self.config, self.predictor_registry, callback
        )

        self.predictor_activator = PredictorActivator(
            self.config, self.predictor_registry, self.context_tracker
        )
        self.predictor_activator.combination_policy = "meritocracy"

    # Predict function - calls the predict on predictor_activator (this calls
    # language model predictions on each of the defined predictor)

    def predict(self):
        multiplier = 1
        (wordprob, word, sentprob, sent) = self.predictor_activator.predict(multiplier)
        if word!=[]:
            #### normalize word probabilites over 10 words. 
            normalized_words = word[0:10]
            prob_sum_over10 = 0.0
            words = []
            for w in enumerate(word):
                prob_sum_over10 += w.probability
                words.append(w.word)

        return (wordprob, [(p.word, p.probability/prob_sum_over10) for p in word], sentprob, [(p.word, p.probability) for p in sent])

    # Paramters updated in ACAT are synced with the parameters initialized 
    # in ConvAssist Predictors.    
    def update_params(self, test_gen_sentence_pred, retrieve_from_AAC):
        self.predictor_activator.update_params(test_gen_sentence_pred, retrieve_from_AAC)


    def read_updated_toxicWords(self):
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
        self.predictor_activator.recreate_canned_phrasesDB()

    """
    This function is called from ACAT --> NamedPipeApplication --> ConvAssist to "learn" a sentence, word or phrase
    """
    def learn_db(self, text):
        sentences = sent_tokenize(text)
        for eachSent in sentences:
            self.predictor_activator.learn_text(eachSent)

    """
    This function is called from NamedPipeApplication --> ConvAssist 
    Checks if models associated with a predictor is loaded
    """
    def check_model(self):
        status = self.predictor_registry.model_status()
        return status

    def close_database(self):
        self.predictor_registry.close_database()