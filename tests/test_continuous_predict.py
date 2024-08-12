import unittest

import configparser
from pathlib import Path

import convAssist.callback
import convAssist

from convAssist.logger import ConvAssistLogger
import logging

#SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = Path(__file__).resolve().parent / "test_data" / "testWordModePredictor"

# Define and create PresageCallback object
class DemoCallback(convAssist.callback.Callback):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def past_stream(self):
        return self.buffer

    def future_stream(self):
        return ""
    def update(self, character):
        if character == "\b" and len(self.buffer) > 0:
            self.buffer[:-1]
        else:
            self.buffer += character
        print("updated stream = ", self.buffer)
        print("past stream = ", self.past_stream())


### config file for shorthand mode
# shorthand_config_file = os.path.join(SCRIPT_DIR, "shortHandMode.ini")
# sh_config = configparser.ConfigParser()
# sh_config.read(shorthand_config_file)

### config file for sentence completion mode
# sentence_config_file = os.path.join(SCRIPT_DIR, "sentenceMode.ini")
# sent_config = configparser.ConfigParser()
# sent_config.read(sentence_config_file)

### config file for word prediction mode
# wordpred_config_file = os.path.join(SCRIPT_DIR, "wordPredMode.ini")
# wordpred_config = configparser.ConfigParser()
# wordpred_config.read(wordpred_config_file)

### config file for CannedPhrases mode
# canned_config_file = os.path.join(SCRIPT_DIR, "cannedPhrasesMode.ini")
# canned_config = configparser.ConfigParser()
# canned_config.read(canned_config_file)



###### Define the shorthand and sentence completion pressagio objects
# callback = DemoCallback("")
# shortHandConvAssist = convAssist.ConvAssist(callback, sh_config)
# sentCompleteConvAssist = convAssist.ConvAssist(callback, sent_config)
# if(sentCompleteConvAssist.check_model()==1):
#     print("SENTENCE COMPLETION MODEL LOADED")
# wordCompleteConvAssist = convAssist.ConvAssist(callback, wordpred_config)
# wordCompleteConvAssist = convAssist.ConvAssist(callback, word_config)
# cannedConvAssist = pressagio.Pressagio(callback, canned_config)


# while (True):
    ##### Activate shorthand predictor: 
    #conv_assist = shortHandConvAssist
    #conv_assist = cannedConvAssist
    #### Or sentence completion Predictor : 
    # conv_assist = wordCompleteConvAssist
    # buffer = input("enter your selection:")
    # conv_assist.callback.update(buffer)
    # prefix = conv_assist.context_tracker.prefix()
    # # context = self.prsg.context()
    # context = conv_assist.context_tracker.past_stream()
    # print("PREFIX = ", prefix, " CONTEXT = ", context)
#    nextLetterProbs, predictions = conv_assist.predict()
#    print(predictions)
    # word_nextLetterProbs, word_predictions , sentence_nextLetterProbs, sentence_predictions = conv_assist.predict()
    # print("word_nextLetterProbs ----", word_nextLetterProbs)
    # print("word_predictions: ----- ", word_predictions)
    # print("sentence_nextLetterProbs ---- ", sentence_nextLetterProbs)
    # print("sentence_predictions: ----- ", sentence_predictions)


    # print("GOING INTO SENTENCE COMPLETION MODE")
    # conv_assist = sentCompleteConvAssist
    # # print("MODELSTATUS = ", conv_assist.check_model())
    # word_nextLetterProbs, word_predictions , sentence_nextLetterProbs, sentence_predictions = conv_assist.predict()
    # print("word_nextLetterProbs ----", word_nextLetterProbs)
    # print("word_predictions: ----- ", word_predictions)
    # print("sentence_nextLetterProbs ---- ", sentence_nextLetterProbs)
    # print("sentence_predictions: ----- ", sentence_predictions)

class TestWordCompleteConvAssist(unittest.TestCase):
    def setUp(self):
        wordpred_config_file = Path(SCRIPT_DIR) / ("wordPredMode.ini")
        wordpred_config = configparser.ConfigParser()
        wordpred_config.read(wordpred_config_file)
        self.callback = DemoCallback("h")
        self.conv_assist = convAssist.ConvAssist(self.callback, wordpred_config)
        
        self.logger = ConvAssistLogger("testWordCompleteConvAssist", "logs", logging.DEBUG)
        self.logger.setLogger()
        

    def tearDown(self) -> None: 
        return super().tearDown()
    
    def test_continuous_predict(self):
        prefix = self.conv_assist.context_tracker.prefix()
        context = self.conv_assist.context_tracker.past_stream()
        word_nextLetterProbs, word_predictions , sentence_nextLetterProbs, sentence_predictions = self.conv_assist.predict()
        # self.logger.debug(f"word_nextLetterProbs ---- {word_nextLetterProbs}")
        assert word_nextLetterProbs ==  [('a', 0.25), ('o', 0.29), ('e', 0.24), ('i', 0.11), ('u', 0.1), ('m', 0.01)]
        # self.logger.debug(f"word_predictions: ----- {word_predictions}")
        assert word_predictions == [('have', 0.37047250193648334), ('how', 0.17189775367931834), ('here', 0.11327652982184351), ('he', 0.0854841208365608), ('help', 0.06193648334624322), ('has', 0.04591789310611928), ('her', 0.044523625096824164), ('had', 0.040836560805577064), ('him', 0.036003098373353984), ('his', 0.029651432997676214)]
        # self.logger.debug(f"sentence_nextLetterProbs ---- {sentence_nextLetterProbs}")
        assert sentence_nextLetterProbs == []
        # self.logger.debug(f"sentence_predictions: ----- {sentence_predictions}")
        assert sentence_predictions == []
