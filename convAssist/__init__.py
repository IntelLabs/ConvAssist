# Copyright (C) <year(s)> Intel Corporation


# SPDX-License-Identifier: Apache-2.0
"""
Init class for ConvAssist Language Model Predictors - config files, predictor registry, are all initialized. 
"""

from glob import glob
from nltk import sent_tokenize
import convAssist.word_sentence_predictor
import convAssist.context_tracker


class ConvAssist:
    def __init__(self, callback, config, dbconnection=None):
        self.config = config
        self.callback = callback

        self.predictor_registry = convAssist.word_sentence_predictor.PredictorRegistry(
            self.config, dbconnection
        )
        self.context_tracker = convAssist.context_tracker.ContextTracker(
            self.config, self.predictor_registry, callback
        )

        self.predictor_activator = convAssist.word_sentence_predictor.PredictorActivator(
            self.config, self.predictor_registry, self.context_tracker
        )
        self.predictor_activator.combination_policy = "meritocracy"

    """
    Predict function - calls the predict on predictor_activator (this calls language model predictions on each of the defined predictor) 
    """

    def predict(self):
        multiplier = 1
        (wordprob, word, sentprob, sent) = self.predictor_activator.predict(multiplier)
        if word!=[]:
            #### normalize word probabilites over 10 words. 
            word = word[0:10]
            prob_sum_over10 = 0.0
            words = []
            for i in range(0, len(word)):
                prob_sum_over10 += word[i].probability
                words.append(word[i].word)

        return (wordprob, [(p.word, p.probability/prob_sum_over10) for p in word], sentprob, [(p.word, p.probability) for p in sent])

    """
    Paramters updated in ACAT are synced with the parameters initialized in ConvAssist Predictors. 
    """

    def update_params(self,test_gen_sentence_pred,retrieve_from_AAC):
        self.predictor_activator.update_params(test_gen_sentence_pred,retrieve_from_AAC)


    def read_updated_toxicWords(self):
        self.predictor_activator.read_updated_toxicWords()

    """
    This function is called from ACAT --> NamedPipeApplication --> ConvAssist to set the Logs location for ConvAssist's logger
    """
    def setLogLocation(self, filename, pathLoc , level):
        self.predictor_activator.set_log(filename, pathLoc, level)

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
