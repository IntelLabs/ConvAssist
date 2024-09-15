# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import configparser
import json
import logging
from pathlib import Path

from ConvAssist.ConvAssist import ConvAssist

SCRIPT_DIR = Path(__file__).resolve().parent.parent

# config file for shorthand mode
shorthand_config_file = Path(SCRIPT_DIR) / ("ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/shortHandMode.ini")
sh_config = configparser.ConfigParser()
sh_config.read(shorthand_config_file)

# config file for sentence completion mode
sentence_config_file = Path(SCRIPT_DIR) / ("ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/sentenceMode.ini")
sent_config = configparser.ConfigParser()
sent_config.read(sentence_config_file)

# config file for word prediction mode
wordpred_config_file = Path(SCRIPT_DIR) / ("ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/wordPredMode.ini")
wordpred_config = configparser.ConfigParser()
wordpred_config.read(wordpred_config_file)

# config file for canned phrases mode
canned_config_file = Path(SCRIPT_DIR) / ("ACAT_ConvAssist_Interface/ConvAssistCPApp/resources/cannedPhrasesMode.ini")
canned_config = configparser.ConfigParser()
canned_config.read(canned_config_file)

###### Define the shorthand ConvAssist objects
shortHandConvAssist = ConvAssist("shorthand", config=sh_config, log_level=logging.INFO)

# # Define the canned ConvAssist objects
# cannedPhrasesConvAssist = ConvAssist("canned", config=canned_config)

# Define the sentence prediction ConvAssist objects
sentCompleteConvAssist = ConvAssist("sentence", config=sent_config, log_level=logging.INFO)
if(sentCompleteConvAssist.check_model()):
    print("SENTENCE COMPLETION MODEL LOADED")

# Define the word ConvAssist objects
wordCompleteConvAssist = ConvAssist("word", config=wordpred_config, log_level=logging.INFO)
if(wordCompleteConvAssist.check_model()):
    print("WORD PREDICTION MODEL LOADED")

convAssists = [wordCompleteConvAssist, sentCompleteConvAssist, shortHandConvAssist]
    
def main():
    while (True):
        buffer = input("Enter text ('close' to exit): ")
        if buffer == "close":
            print("Closing CLI.")
            break
            
        print("GOING INTO WORD PREDICTION MODE")

        for convAssist in convAssists:
            convAssist.context_tracker.context = buffer
            prefix = convAssist.context_tracker.token(0)
            context = convAssist.context_tracker.context
            print("PREFIX = ", prefix, " CONTEXT = ", context)

            word_nextLetterProbs, \
                word_predictions, \
                    sentence_nextLetterProbs, \
                    sentence_predictions = convAssist.predict()
            
            print("word_nextLetterProbs ----", json.dumps(word_nextLetterProbs))
            print("word_predictions: ----- ", json.dumps(word_predictions))
            print("sentence_nextLetterProbs ---- ", json.dumps(sentence_nextLetterProbs))
            print("sentence_predictions: ----- ", json.dumps(sentence_predictions))
            print("---------------------------------------------------")

if __name__ == "__main__":
    main()