# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import configparser
from pathlib import Path

from ConvAssist import ConvAssist
from ConvAssist.utilities.callback import BufferedCallback

SCRIPT_DIR = Path(__file__).resolve().parent

class CPCallback(BufferedCallback):
    def update(self, character):
        super().update(character)
        print("updated stream = ", self.buffer)
        print("past stream = ", self.past_stream())

# config file for shorthand mode
shorthand_config_file = Path(SCRIPT_DIR) / ("resources/shortHandMode.ini")
sh_config = configparser.ConfigParser()
sh_config.read(shorthand_config_file)

# config file for sentence completion mode
sentence_config_file = Path(SCRIPT_DIR) / ("resources/sentenceMode.ini")
sent_config = configparser.ConfigParser()
sent_config.read(sentence_config_file)

# config file for word prediction mode
wordpred_config_file = Path(SCRIPT_DIR) / ("resources/wordPredMode.ini")
wordpred_config = configparser.ConfigParser()
wordpred_config.read(wordpred_config_file)

# config file for canned phrases mode
canned_config_file = Path(SCRIPT_DIR) / ("resources/cannedPhrasesMode.ini")
canned_config = configparser.ConfigParser()
canned_config.read(canned_config_file)

###### Define the shorthand ConvAssist objects
callback = CPCallback("")
shortHandConvAssist = ConvAssist(callback, sh_config)

# Define the word prediction and sentence completion ConvAssist objects
sentCompleteConvAssist = ConvAssist(callback, sent_config)
if(sentCompleteConvAssist.check_model()==1):
    print("SENTENCE COMPLETION MODEL LOADED")

wordCompleteConvAssist = ConvAssist(callback, wordpred_config)
if(wordCompleteConvAssist.check_model()==1):
    print("WORD PREDICTION MODEL LOADED")
    
def main():
    while (True):
        ConvAssist = wordCompleteConvAssist
        buffer = input("Enter text ('close' to exit): ")
        if buffer == "close":
            print("Closing CLI.")
            break
        ConvAssist.callback.update(buffer)
        prefix = ConvAssist.context_tracker.prefix()
        context = ConvAssist.context_tracker.past_stream()
        print("PREFIX = ", prefix, " CONTEXT = ", context)
        word_nextLetterProbs, word_predictions , sentence_nextLetterProbs, sentence_predictions = ConvAssist.predict()
        print("word_nextLetterProbs ----", word_nextLetterProbs)
        print("word_predictions: ----- ", word_predictions)
        print("sentence_nextLetterProbs ---- ", sentence_nextLetterProbs)
        print("sentence_predictions: ----- ", sentence_predictions)

        print("GOING INTO SENTENCE COMPLETION MODE")
        ConvAssist = sentCompleteConvAssist
        word_nextLetterProbs, word_predictions , sentence_nextLetterProbs, sentence_predictions = ConvAssist.predict()
        print("word_nextLetterProbs ----", word_nextLetterProbs)
        print("word_predictions: ----- ", word_predictions)
        print("sentence_nextLetterProbs ---- ", sentence_nextLetterProbs)
        print("sentence_predictions: ----- ", sentence_predictions)

if __name__ == "__main__":
    main()