
"""
 Copyright (C) 2023 Intel Corporation

 SPDX-License-Identifier: Apache-2.0

   Systray application for ConvAssist (word predictor)

"""
import configparser
import glob
import os
import ConvAssist
import time
import win32pipe
import win32file
import pywintypes
import json
import threading
import sys
import pystray
import PIL.Image
import psutil
from pystray import MenuItem as item
from tkinter import *
from PIL import Image, ImageTk
from Messages import *
import ConvAssistUtilities
from enum import IntEnum
from ConvAssist.utilities.logging import *
from datetime import datetime
import shutil
from ConvAssist import callback
import logging

global kill_ConvAssist
global ConvAssist_callback
global breakMainLoop
global clientConnected
global windowOpened
global licenseWindow
global word_config_set
global sh_config_set
global sent_config_set
global sent_config_change
global canned_config_set
global word_suggestions
global path_logs
global enable_logs
global ConvAssistLog
global icon_logo
global conv_normal
global conv_shorthand
global conv_sentence
global conv_canned_phrases

conv_normal = None
conv_shorthand = None
conv_sentence = None
conv_canned_phrases = None

kill_ConvAssist = False
icon_logo = None
ConvAssist_callback = None
breakMainLoop = False
clientConnected = False
windowOpened = False
licenseWindow = False
word_config_set = False
sh_config_set = False
sent_config_set = False
canned_config_set = False
sent_config_change = False
word_suggestions = 15
path_logs = ""
enable_logs = True
ConvAssistLog = ConvAssistLogger("ConvAssist_Pred", "", logging.INFO)
ConvAssistLog.setLogger()

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
image = PIL.Image.open(os.path.join(SCRIPT_DIR, "Assets", "icon_tray.png"))
pipeName = "ACATConvAssistPipe"
retries = 10



class ConvAssistMessageTypes(IntEnum):
    NONE = 0
    SETPARAM = 1
    NEXTWORDPREDICTION = 2
    NEXTWORDPREDICTIONRESPOSNE = 3
    NEXTSENTENCEPREDICTION = 4
    NEXTSENTENCEPREDICTIONRESPOSNE = 5
    LEARNWORDS = 6
    LEARNCANNED = 7
    LEARNSHORTHAND = 8
    LEARNSENTENCES = 9
    FORCEQUITAPP = 10


class ConvAssistPredictionTypes(IntEnum):
    NONE = 0
    NORMAL = 1
    SHORTHANDMODE = 2
    CANNEDPHRASESMODE = 3
    SENTENCES = 4


class ParameterType(IntEnum):
    NONE = 0
    PATH = 1
    SUGGESTIONS = 2
    TESTGENSENTENCEPRED = 3
    RETRIEVEAAC = 4
    PATHSTATIC = 5
    PATHPERSONALIZED = 6
    PATHLOG = 7
    ENABLELOGS = 8


class DemoCallback(ConvAssist.utilities.callback.Callback):
    """
    Define and create ConvAssist Callback object
    """

    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def past_stream(self):
        return self.buffer

    def future_stream(self):
        return ""

    def update(self, text):
        self.buffer = text


def addTextToWindow(text):
    """Add text to the UI window

    Args:
        text (string): Text to be displayed in the UI
    """
    global windowOpened
    try:
        if windowOpened:
            text_box.configure(state='normal')
            text_box.insert(END," " + text)
            text_box.configure(state='disabled')
    except Exception as e:
        ConvAssistLog.exception(f"Exception: {e} Value: {text}")


def threaded_function(Pipehandle, retries):
    """
    Main Thread, called when a server was found: Receives and send messages, Event message terminate App

    :param Pipehandle: Handle of the pipe
    :param retries:
    :return: none
    """
    global ConvAssistLog
    global word_suggestions
    global word_config_set
    global sent_config_set
    global canned_config_set
    global sh_config_set
    global sent_config_change
    global path_logs
    global enable_logs
    global conv_normal
    global conv_shorthand
    global conv_sentence
    global conv_canned_phrases
    
    sentences_count = 0
    retrieve_from_AAC = False #Default value
    test_gen_sentence_pred = False #Default value
    path_static = ""
    path_personalized = ""
    prediction_type = ConvAssistPredictionTypes.NORMAL
    word_prediction = []
    next_Letter_Probs = []
    sentence_nextLetterProbs = []
    sentence_predictions = []
    ConvAssist_callback = DemoCallback("")
    breakLoop = False
    counter = 0
    string_resultSentence = ""
    ConvAssistLog.info("Connection with ACAT established")
    global clientConnected
    clientConnected = True
    addTextToWindow("*** Successful connection ***\n")
    while not breakLoop:
        try:
            res = win32pipe.SetNamedPipeHandleState(Pipehandle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            if res == 0:
                print(f"SetNamedPipeHandleState return code: {res}")
            while not breakLoop:
                msg = ''
                (resp, data) = win32file.ReadFile(Pipehandle, 1, pywintypes.OVERLAPPED())
                while resp == 234:
                    msg = msg + bytes(data).decode('ASCII')
                    resp, data = win32file.ReadFile(Pipehandle, 1, pywintypes.OVERLAPPED())
                if resp == 0:  # end of stream is reached
                    msg = msg + bytes(data).decode('ASCII')
                try:
                    jsonstring = json.loads(msg)
                    messageReceived = ConvAssistMessage.jsonDeserialize(jsonstring)
                except Exception as e:
                    ConvAssistLog.exception(f" jsonDeserialize incomming message {msg}  {e}")
                    addTextToWindow(f"Exception jsonDeserialize incomming message {msg}  {e} \n")
                    PredictionResponse = WordAndCharacterPredictionResponse(
                        ConvAssistMessageTypes.NONE,
                        ConvAssistPredictionTypes.NONE,
                        "",
                        "",
                        "",
                        "")
                    json_string_result = json.dumps(PredictionResponse.__dict__)
                    string_result = str(json_string_result).encode()
                    win32file.WriteFile(Pipehandle, string_result)
                    ConvAssistLog.info("Exception message send")
                match messageReceived.MessageType:
                    case ConvAssistMessageTypes.NONE:
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.NONE,
                            ConvAssistPredictionTypes.NONE,
                            "",
                            "",
                            "",
                            "")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)

                    case ConvAssistMessageTypes.SETPARAM:
                        try:
                            jsonstringparams = json.loads(messageReceived.Data)
                            params = ConvAssistSetParam.jsonDeserialize(jsonstringparams)
                            match params.Parameter:
                                case ParameterType.PATH:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    setModelsParameters(params.Value,test_gen_sentence_pred,retrieve_from_AAC,path_static,path_personalized)
                                    if word_config_set:
                                        try:
                                            conv_normal = ConvAssist.ConvAssist(ConvAssist_callback, word_config)
                                            ConvAssistLog.info("conv_normal set")
                                            addTextToWindow("conv_normal set \n")
                                            if enable_logs:
                                                conv_normal.setLogLocation(getDate() + "_NORMAL", path_logs, logging.INFO)
                                            else:
                                                conv_normal.setLogLocation(getDate() + "_NORMAL", path_logs, logging.ERROR)
                                        except Exception as exception_Normal:
                                            ConvAssistLog.exception(f" setting predictor object Normal: received message {exception_Normal}")
                                            addTextToWindow(f"Exception setting predictor object Normal: received message {exception_Normal} \n")
                                            word_config_set = False
                                    if sh_config_set:
                                        try:
                                            conv_shorthand = ConvAssist.ConvAssist(ConvAssist_callback, sh_config)
                                            ConvAssistLog.info("conv_shorthand set")
                                            addTextToWindow("conv_shorthand set \n")
                                            if enable_logs:
                                                conv_shorthand.setLogLocation(getDate() + "_SHORTHAND", path_logs, logging.INFO)
                                            else:
                                                conv_shorthand.setLogLocation(getDate() + "_SHORTHAND", path_logs, logging.ERROR)
                                        except Exception as exception_shorthand:
                                            ConvAssistLog.exception(f" setting predictor object shorthand: received message {exception_shorthand}")
                                            addTextToWindow(f"Exception setting predictor object shorthand: received message {exception_shorthand} \n")
                                            sh_config_set = False
                                    if sent_config_set:
                                        try:
                                            conv_sentence = ConvAssist.ConvAssist(ConvAssist_callback, sent_config)
                                            ConvAssistLog.info("conv_sentence set")
                                            conv_sentence.read_updated_toxicWords()
                                            ConvAssistLog.info("reading updated toxic words from personalized text file")
                                            addTextToWindow("conv_sentence set \n")
                                            if sent_config_change:
                                                ConvAssistLog.info("Sentence INI NEW config detected")
                                                addTextToWindow("Sentence INI NEW config detected \n")
                                                conv_sentence.update_params(str(test_gen_sentence_pred), str(retrieve_from_AAC))
                                                sent_config_change = False
                                            else:
                                                ConvAssistLog.info("Sentence INI no new modifications")
                                                addTextToWindow("Sentence INI NO new modifications \n")
                                            if enable_logs:
                                                conv_sentence.setLogLocation(getDate() + "_SENTENCE", path_logs, logging.INFO)
                                            else:
                                                conv_sentence.setLogLocation(getDate() + "_SENTENCE", path_logs, logging.ERROR)
                                        except Exception as exception_Sentence:
                                            ConvAssistLog.exception(f" setting predictor object Sentence: received message {exception_Sentence}")
                                            addTextToWindow(f"Exception setting predictor object Sentence: received message {exception_Sentence} \n")
                                            sent_config_set = False
                                    if canned_config_set:
                                        try:
                                            conv_canned_phrases = ConvAssist.ConvAssist(ConvAssist_callback, canned_config)
                                            conv_canned_phrases.cannedPhrase_recreateDB()
                                            ConvAssistLog.info("conv_canned_phrases set")
                                            addTextToWindow("conv_canned_phrases set \n")
                                            if enable_logs:
                                                conv_canned_phrases.setLogLocation(getDate() + "_CANNED", path_logs, logging.INFO)
                                            else:
                                                conv_canned_phrases.setLogLocation(getDate() + "_CANNED", path_logs, logging.ERROR)
                                        except Exception as exception_canned_phrases:
                                            ConvAssistLog.exception(f" setting predictor object canned phrases: received message {exception_canned_phrases}")
                                            addTextToWindow(f"Exception setting predictor object canned phrases: received message {exception_canned_phrases} \n")
                                            canned_config_set = False
                                case ParameterType.SUGGESTIONS:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        word_suggestions = int(params.Value)
                                    except Exception as e:
                                        ConvAssistLog.exception(f" Parameter suggestions value {params.Value} message {e}")
                                        addTextToWindow(f"Exception Parameter suggestions value {params.Value} message {e} \n")
                                        word_suggestions = 15
                                case ParameterType.TESTGENSENTENCEPRED:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        test_gen_sentence_pred = True if params.Value.lower() == "true" else False
                                    except Exception as e:
                                        ConvAssistLog.exception(f" Parameter suggestions value {params.Value} message {e}")
                                        addTextToWindow(f"Exception Parameter suggestions value {params.Value} message {e} \n")
                                        test_gen_sentence_pred = False
                                case ParameterType.RETRIEVEAAC:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        retrieve_from_AAC = True if params.Value.lower() == "true" else False
                                    except Exception as e:
                                        ConvAssistLog.exception(f" Parameter suggestions value {params.Value} message {e}")
                                        addTextToWindow(f"Exception Parameter suggestions value {params.Value} message {e} \n")
                                        retrieve_from_AAC = False
                                case ParameterType.PATHSTATIC:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        path_static = params.Value
                                    except Exception as e:
                                        ConvAssistLog.exception(f" Parameter path static value {params.Value} message {e}")
                                        addTextToWindow(f"Exception Parameter path static value {params.Value} message {e} \n")
                                        path_static = ""
                                case ParameterType.PATHPERSONALIZED:
                                    ConvAssistLog.debug(f"Parameter type: {params.Parameter} Value received {params.Value}")
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        path_personalized = params.Value
                                    except Exception as e:
                                        ConvAssistLog.exception(f" Parameter path personalized value {params.Value} message {e}")
                                        addTextToWindow(f"Exception Parameter path personalized value {params.Value} message {e} \n")
                                        path_personalized = ""
                                case ParameterType.PATHLOG:
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        path_logs = params.Value
                                        if enable_logs:
                                            if ConvAssistLog.IsLogInitialized():
                                                ConvAssistLog.Close()
                                            ConvAssistLog = None
                                            ConvAssistLog = ConvAssistLogger(getDate() + "_MAIN", path_logs, logging.INFO)
                                            ConvAssistLog.setLogger()
                                            ConvAssistLog.debug(f"Log created in: {path_logs}")
                                    except Exception as e:
                                        addTextToWindow(f"Exception Parameter path personalized value {path_logs} message {e} \n")
                                case ParameterType.ENABLELOGS:
                                    addTextToWindow(f"Parameter type: {params.Parameter} Value received {params.Value} \n")
                                    try:
                                        enable_logs = True if params.Value.lower() == "true" else False
                                    except Exception as e:
                                        addTextToWindow(f"Exception Parameter enable logs value {params.Value} message {e} \n")
                        except Exception as e:
                            ConvAssistLog.exception(f" Parameters received message {e}")
                            addTextToWindow(f"Exception Parameters received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.SETPARAM,
                            ConvAssistPredictionTypes.NONE,
                            "",
                            "",
                            "",
                            "")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)
                        ConvAssistLog.info("Parameters message answered")
                        addTextToWindow("Parameters message answered \n")

                    case ConvAssistMessageTypes.NEXTWORDPREDICTION:
                        ConvAssistLog.debug(f"Prediction requested type: {messageReceived.PredictionType} Prediction Message: {messageReceived.Data}")
                        addTextToWindow(f"Prediction requested type: {messageReceived.PredictionType} Prediction Message: {messageReceived.Data} \n")
                        match messageReceived.PredictionType:
                            case ConvAssistPredictionTypes.NONE:
                                word_prediction = []
                                next_Letter_Probs = []
                                sentence_nextLetterProbs = []
                                sentence_predictions = []
                                prediction_type = ConvAssistPredictionTypes.NONE

                            case ConvAssistPredictionTypes.NORMAL:
                                if word_config_set:
                                    try:
                                        sentences_count = 0
                                        count = len(messageReceived.Data)
                                        if count == 1 and messageReceived.Data.isspace():
                                            conv_normal.callback.update("")
                                        else:
                                            conv_normal.callback.update(messageReceived.Data)
                                        conv_normal.context_tracker.prefix()
                                        conv_normal.context_tracker.past_stream()
                                        next_Letter_Probs, word_prediction, sentence_nextLetterProbs, sentence_predictions = conv_normal.predict()
                                        prediction_type = ConvAssistPredictionTypes.NORMAL
                                    except Exception as e:
                                        ConvAssistLog.exception(f" ConvAssistPredictionTypes.NORMAL: {e}")
                                        addTextToWindow(f"Exception ConvAssistPredictionTypes.NORMAL: {e} \n")
                                        if len(word_prediction) == 0:
                                            word_prediction = []
                                        if len(next_Letter_Probs) == 0:
                                            next_Letter_Probs = []
                                        sentence_nextLetterProbs = []
                                        sentence_predictions = []
                                        prediction_type = ConvAssistPredictionTypes.NORMAL

                            case ConvAssistPredictionTypes.SHORTHANDMODE:
                                if sh_config_set:
                                    try:
                                        sentences_count = 0
                                        conv_shorthand.callback.update(messageReceived.Data)
                                        conv_shorthand.context_tracker.prefix()
                                        conv_shorthand.context_tracker.past_stream()
                                        next_Letter_Probs, word_prediction, sentence_nextLetterProbs, sentence_predictions = conv_shorthand.predict()
                                        prediction_type = ConvAssistPredictionTypes.SHORTHANDMODE
                                    except Exception as e:
                                        ConvAssistLog.exception(f" ConvAssistPredictionTypes.SHORTHANDMODE: {e}")
                                        addTextToWindow(f"Exception ConvAssistPredictionTypes.SHORTHANDMODE: {e} \n")
                                        word_prediction = []
                                        next_Letter_Probs = []
                                        sentence_nextLetterProbs = []
                                        sentence_predictions = []
                                        prediction_type = ConvAssistPredictionTypes.SHORTHANDMODE

                            case ConvAssistPredictionTypes.CANNEDPHRASESMODE:
                                if canned_config_set:
                                    try:
                                        sentences_count = 6
                                        conv_canned_phrases.callback.update(messageReceived.Data)
                                        conv_canned_phrases.context_tracker.prefix()
                                        conv_canned_phrases.context_tracker.past_stream()
                                        next_Letter_Probs, word_prediction, sentence_nextLetterProbs, sentence_predictions = conv_canned_phrases.predict()
                                        prediction_type = ConvAssistPredictionTypes.CANNEDPHRASESMODE
                                    except Exception as e:
                                        ConvAssistLog.exception(f" ConvAssistPredictionTypes.CANNEDPHRASESMODE: {e}")
                                        addTextToWindow(f"Exception ConvAssistPredictionTypes.CANNEDPHRASESMODE: {e} \n")
                                        word_prediction = []
                                        next_Letter_Probs = []
                                        sentence_nextLetterProbs = []
                                        sentence_predictions = []
                                        prediction_type = ConvAssistPredictionTypes.CANNEDPHRASESMODE

                        next_Letter_Probs = sort_List(next_Letter_Probs, 20)
                        word_prediction = sort_List(word_prediction, 10)
                        sentence_nextLetterProbs = sort_List(sentence_nextLetterProbs, 0)
                        sentence_predictions = sort_List(sentence_predictions, sentences_count)
                        result_Letters = str(next_Letter_Probs)
                        result_Words = str(word_prediction)
                        result_Letters_Sentence = str(sentence_nextLetterProbs)
                        result_Sentences = str(sentence_predictions)
                        resultAll = result_Words + "/" + result_Letters
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.NEXTWORDPREDICTIONRESPOSNE,
                            prediction_type,
                            result_Words,
                            result_Letters,
                            result_Letters_Sentence,
                            result_Sentences)
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)

                    case ConvAssistMessageTypes.NEXTSENTENCEPREDICTION:
                        ConvAssistLog.debug(f"Prediction requested type: {messageReceived.MessageType} Prediction Message: {messageReceived.Data}")
                        addTextToWindow(f"Prediction requested type: {messageReceived.MessageType} Prediction Message: {messageReceived.Data} \n")
                        if sent_config_set:
                            status_model = conv_sentence.check_model()
                        if sent_config_set and status_model == 1:
                            try:
                                conv_sentence.callback.update(messageReceived.Data)
                                conv_sentence.context_tracker.prefix()
                                conv_sentence.context_tracker.past_stream()
                                next_Letter_Probs, word_prediction, sentence_nextLetterProbs, sentence_predictions = conv_sentence.predict()
                                prediction_type = ConvAssistPredictionTypes.NONE
                            except Exception as e:
                                ConvAssistLog.exception(f" ConvAssistPredictionTypes.SENTENCEMODE: {e}")
                                addTextToWindow(f"Exception ConvAssistPredictionTypes.SENTENCEMODE: {e} \n")
                                word_prediction = []
                                next_Letter_Probs = []
                                sentence_nextLetterProbs = []
                                sentence_predictions = []
                                prediction_type = ConvAssistPredictionTypes.NONE

                        next_Letter_Probs = sort_List(next_Letter_Probs, 0)
                        word_prediction = sort_List(word_prediction, 0)
                        sentence_nextLetterProbs = sort_List(sentence_nextLetterProbs, 0)
                        sentence_predictions = sort_List(sentence_predictions, 6)
                        result_Letters = str(next_Letter_Probs)
                        result_Words = str(word_prediction)
                        result_Letters_Sentence = str(sentence_nextLetterProbs)
                        result_Sentences = str(sentence_predictions)
                        resultAll = result_Sentences
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.NEXTSENTENCEPREDICTIONRESPOSNE,
                            prediction_type,
                            result_Words,
                            result_Letters,
                            result_Letters_Sentence,
                            result_Sentences)
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_resultSentence = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_resultSentence)
                    case ConvAssistMessageTypes.LEARNWORDS:
                        try:
                            ConvAssistLog.debug(f"Learn for WORDS/NORMAL mode: Data - {messageReceived.Data}")
                            addTextToWindow(f"Learn for WORDS/NORMAL mode: Data - {messageReceived.Data} \n")
                            conv_normal.learn_db(messageReceived.Data)
                        except Exception as e:
                            ConvAssistLog.exception(f" Learn received message {e}")
                            addTextToWindow(f"Exception Learn received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.LEARNWORDS,
                            ConvAssistPredictionTypes.NONE,"","","","")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)
                        ConvAssistLog.info("Learn message answered")
                        addTextToWindow("Learn message answered \n")
                    case ConvAssistMessageTypes.LEARNCANNED:
                        try:
                            ConvAssistLog.debug(f"Learn for CANNEDPHRASESMODE mode: Data - {messageReceived.Data}")
                            addTextToWindow(f"Learn for CANNEDPHRASESMODE mode: Data - {messageReceived.Data} \n")
                            conv_canned_phrases.learn_db(messageReceived.Data)
                        except Exception as e:
                            ConvAssistLog.exception(f" Learn received message {e}")
                            addTextToWindow(f"Exception Learn received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.LEARNCANNED,
                            ConvAssistPredictionTypes.NONE,"","","","")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)
                        ConvAssistLog.info("Learn message answered")
                        addTextToWindow("Learn message answered \n")
                    case ConvAssistMessageTypes.LEARNSHORTHAND:
                        try:
                            ConvAssistLog.debug(f"Learn for SHORTHANDMODE mode: Data - {messageReceived.Data}")
                            addTextToWindow(f"Learn for SHORTHANDMODE mode: Data - {messageReceived.Data} \n")
                            conv_shorthand.learn_db(messageReceived.Data)
                        except Exception as e:
                            ConvAssistLog.exception(f" Learn received message {e}")
                            addTextToWindow(f"Exception Learn received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.LEARNSHORTHAND,
                            ConvAssistPredictionTypes.NONE,"","","","")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)
                        ConvAssistLog.info("Learn message answered")
                        addTextToWindow("Learn message answered \n")
                    case ConvAssistMessageTypes.LEARNSENTENCES:
                        try:
                            ConvAssistLog.debug(f"Learn for SENTENCES mode: Data - {messageReceived.Data}")
                            addTextToWindow(f"Learn for SENTENCES mode: Data - {messageReceived.Data} \n")
                            conv_sentence.learn_db(messageReceived.Data)
                        except Exception as e:
                            ConvAssistLog.exception(f" Learn received message {e}")
                            addTextToWindow(f"Exception Learn received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.LEARNSENTENCES,
                            ConvAssistPredictionTypes.NONE,"","","","")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)
                        ConvAssistLog.info("Learn message answered")
                        addTextToWindow("Learn message answered \n")
                    case ConvAssistMessageTypes.FORCEQUITAPP:
                        try:
                            PredictionResponse = WordAndCharacterPredictionResponse(
                                ConvAssistMessageTypes.NONE,
                                ConvAssistPredictionTypes.NONE,"","","","")
                            json_string_result = json.dumps(PredictionResponse.__dict__)
                            string_result = str(json_string_result).encode()
                            win32file.WriteFile(Pipehandle, string_result)
                            breakLoop = True
                            global kill_ConvAssist
                            kill_ConvAssist = True
                        except Exception as e:
                            ConvAssistLog.exception(f" quit App request {e}")
                    case _:
                        try:
                            ConvAssistLog.info("No type Match message answered")
                            addTextToWindow("No type Match message answered \n")
                        except Exception as e:
                            ConvAssistLog.exception(f" Default, received message {e}")
                            addTextToWindow(f"Exception Default, received message {e} \n")
                        PredictionResponse = WordAndCharacterPredictionResponse(
                            ConvAssistMessageTypes.NONE,
                            ConvAssistPredictionTypes.NONE,"","","","")
                        json_string_result = json.dumps(PredictionResponse.__dict__)
                        string_result = str(json_string_result).encode()
                        win32file.WriteFile(Pipehandle, string_result)


        except pywintypes.error as e:
            ConvAssistLog.exception(f" in main Thread: {e}")
            addTextToWindow(f"Exception in main Thread: {e} \n")
            if e.args[0] == 2:
                if counter > retries:
                    breakLoop = True
                time.sleep(1)
                counter += 1
            elif e.args[0] == 109:
                breakLoop = True


def getDate():
    """Gets the current date

    Returns:
        string: MM-DD-YYYY__HH-MM-SS
    """
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y___%H-%M-%S")
    return "ConvAssist_log_" + date_time


def setPipeClient(PipeServerName, retries):
    """
    Set the Pipe as Client

    :param PipeServerName: Name of the pipe
    :param retries: Amount of retires to connection
    :return: none
    """
    addTextToWindow("      - Establishing connection \n")
    pipeName = f'\\\\.\pipe\{PipeServerName}'
    global clientConnected
    clientConnected = False
    handle = win32file.CreateFile(
        pipeName,
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0,
        None,
        win32file.OPEN_EXISTING,
        0,
        None
    )
    thread = threading.Thread(target=threaded_function, args=(handle, retries))
    thread.start()
    thread.join()


def InitPredict():
    """
    Initialization of the next character prediction
    """
    global windowOpened
    while not breakMainLoop:
        try:
            if kill_ConvAssist:
                print("Kill app active")
                quitapp()
            else:
                # print("      - Setting Connection")
                addTextToWindow("      - Setting Connection \n")
                setPipeClient(pipeName, retries)
        except Exception:
            time.sleep(5)
            addTextToWindow("  An error occurred, no connection established  \n")
            if breakMainLoop:
                ConvAssistLog.info("Closing .exe")
                sys.exit()


def quit_window(icon, item):
    """
    Function for quit the window

    :param icon:
    :param item:
    :return:
    """
    # Necessary to close the App that there is no server connected
    if not clientConnected:
        global breakMainLoop
        breakMainLoop = True
        icon.stop()
        ws.destroy()


def quitapp():
    """
        Force to quit and exit the App
    """
    global breakMainLoop
    breakMainLoop = True
    icon_logo.stop()
    ws.quit()
    # os._exit(1)
    sys.exit()


def show_window(icon, item):
    """
    Function to show the window again

    :param icon:
    :param item:
    :return:
    """
    global windowOpened
    windowOpened = True
    icon.stop()
    ws.after(0, ws.deiconify())


def hide_window():
    """
    Hide the window and show on the system taskbar

    :return:
    """
    if not licenseWindow:
        global windowOpened
        global icon_logo
        windowOpened = False
        text_box.configure(state='normal')
        text_box.delete(1.0, END)
        text_box.configure(state='disabled')
        ws.withdraw()
        menu = (item('Exit', quit_window), item('More info', show_window))
        icon = pystray.Icon("name", image, "ConvAssist", menu)
        icon_logo = icon
        icon.run()


def move_App(e):
    """
    Let the window app move freely

    :param e: event
    :return: void
    """
    ws.geometry(f'+{e.x_root}+{e.y_root}')


def show_license():
    """
    Shows the license text

    :return: void
    """
    global licenseWindow
    if not licenseWindow:
        licenseWindow = True
        label_licence.lift()
        back_button.lift()
        clear_button.lower()
        exit_button.lower()
        license_button.lower()
        label2.lower()


def hide_license():
    """
    Hide the license text

    :return: void
    """
    global licenseWindow
    if licenseWindow:
        licenseWindow = False
        label_licence.lower()
        back_button.lower()
        clear_button.lift()
        exit_button.lift()
        license_button.lift()
        label2.lift()


def clear_textWindow():
    """
    Clears the tex box

    :return: void
    """
    global licenseWindow
    if not licenseWindow:
        text_box.configure(state='normal')
        text_box.delete(1.0, END)
        text_box.configure(state='disabled')


def findProcessIdByName(process_name):
    """
    Get a list of all the PIDs of all the running process whose name contains
    the given string processName

    :param process_name: Name of process to look
    :return: True if process is running
    """
    listOfProcessObjects = []
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            # Check if process name contains the given name string.
            if process_name.lower() in pinfo['name'].lower():
                listOfProcessObjects.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    ConvAssistLog.debug(f"Number of processes with name ConvAssistCPApp.exe: {len(listOfProcessObjects)}")
    if len(listOfProcessObjects) > 2:
        return True
    return False


def setModelsParameters(path_database, test_gen_sentence_prediction, retrieve_AAC,path_static,path_personalized):
    """
    Sets the path to the objects used as a configuration for the predictors

    :param path_database: path of the directories of the databases
    :return: NA
    """
    try:
        """
        Word predictions config file
        """
        global word_config
        global word_config_set
        ConvAssistLog.debug(f"INI file static Path: {path_static}")
        addTextToWindow(f"INI file static Path: {path_static} \n")
        ConvAssistLog.debug(f"INI file Personalized Path: {path_personalized}")
        addTextToWindow(f"INI file Personalized Path: {path_personalized} \n")
        config_file = os.path.join(path_database, "wordPredMode.ini")
        word_config = configparser.ConfigParser()
        file_size = len(word_config.read(config_file))
        if file_size > 0:
            word_config_set = True
            ConvAssistLog.info("INI file for Word pred mode found")
            addTextToWindow("INI file for Word pred mode found \n")
            word_config.set('Selector', 'suggestions', str(word_suggestions))
            ConvAssistLog.debug(f"INI file modification for suggestions of {word_suggestions}")
            addTextToWindow(f"INI file modification for suggestions of {word_suggestions} \n")
            if len(path_static) > 1 and "\\" in path_static and len(path_personalized) > 1 and "\\" in path_personalized:
                word_config.set('DefaultSmoothedNgramPredictor', 'static_resources_path', path_static)
                word_config.set('DefaultSmoothedNgramPredictor', 'personalized_resources_path', path_personalized)
                word_config.set('PersonalSmoothedNgramPredictor', 'static_resources_path', path_static)
                word_config.set('PersonalSmoothedNgramPredictor', 'personalized_resources_path', path_personalized)
                word_config.set('SpellCorrectPredictor', 'static_resources_path', path_static)
            with open(config_file, 'w') as configfile:
                word_config.write(configfile)
        else:
            ConvAssistLog.info("INI file for Word pred mode NOT found")
            addTextToWindow("INI file for Word pred mode NOT found \n")
    except Exception as e:
        word_config_set = False
        ConvAssistLog.exception(f" INI file for Word pred {e}")
        addTextToWindow(f"Exception INI file for Word pred {e} \n")
    try:
        """
        config file for shorthand mode
        """
        global sh_config
        global sh_config_set
        shorthand_config_file = os.path.join(path_database, "shortHandMode.ini")
        sh_config = configparser.ConfigParser()
        file_size = len(sh_config.read(shorthand_config_file))
        if file_size > 0:
            sh_config_set = True
            ConvAssistLog.info("INI file for shorthand mode found")
            addTextToWindow("INI file for shorthand mode found \n")
            if len(path_static) > 1 and "\\" in path_static and len(path_personalized) > 1 and "\\" in path_personalized:
                sh_config.set('ShortHandPredictor', 'static_resources_path', path_static)
                sh_config.set('ShortHandPredictor', 'personalized_resources_path', path_personalized)
            with open(shorthand_config_file, 'w') as configfile:
                sh_config.write(configfile)
        else:
            ConvAssistLog.info("INI file for shorthand mode NOT found")
            addTextToWindow("INI file for shorthand mode NOT found \n")
    except Exception as e:
        sh_config_set = False
        ConvAssistLog.exception(f" INI file for shorthand {e}")
        addTextToWindow(f"Exception INI file for shorthand {e} \n")
    try:
        """
        config file for sentence completion mode
        """
        global sent_config
        global sent_config_set
        global sent_config_change
        sentence_config_file = os.path.join(path_database, "sentenceMode.ini")
        sent_config = configparser.ConfigParser()
        file_size = len(sent_config.read(sentence_config_file))
        if file_size > 0:
            sent_config_set = True
            ConvAssistLog.info("INI file for sentence completion mode found")
            addTextToWindow("INI file for sentence completion mode found \n")
            value_test_gen_sentence_prediction = sent_config.get('SentenceCompletionPredictor', 'test_generalSentencePrediction')
            if (value_test_gen_sentence_prediction.lower() == "true") != test_gen_sentence_prediction:
                sent_config_change = True
                sent_config.set('SentenceCompletionPredictor', 'test_generalSentencePrediction', str(test_gen_sentence_prediction))
                ConvAssistLog.debug(f"INI file modification for test_generalSentencePrediction as {test_gen_sentence_prediction}")
                addTextToWindow(f"INI file modification for test_generalSentencePrediction as {test_gen_sentence_prediction} \n")

            value_retrieve_AAC = sent_config.get('SentenceCompletionPredictor', 'retrieveAAC')
            if (value_retrieve_AAC.lower() == "true") != retrieve_AAC:
                sent_config_change = True
                sent_config.set('SentenceCompletionPredictor', 'retrieveAAC', str(retrieve_AAC))
                ConvAssistLog.debug(f"INI file modification for retrieveAAC as {retrieve_AAC}")
                addTextToWindow(f"INI file modification for retrieveAAC as {retrieve_AAC} \n")
            if len(path_static) > 1 and "\\" in path_static and len(path_personalized) > 1 and "\\" in path_personalized:
                sent_config.set('SentenceCompletionPredictor', 'static_resources_path', path_static)
                sent_config.set('SentenceCompletionPredictor', 'personalized_resources_path', path_personalized)
            with open(sentence_config_file, 'w') as configfileSentence:
                sent_config.write(configfileSentence)
        else:
            ConvAssistLog.info("INI file for sentence completion mode NOT found")
            addTextToWindow("INI file for sentence completion mode NOT found \n")
    except Exception as e:
        sent_config_set = False
        ConvAssistLog.exception(f" INI file for sentence {e}")
        addTextToWindow(f"Exception INI file for sentence {e} \n")

    try:
        """
        config file for CannedPhrases mode
        """
        global canned_config
        global canned_config_set
        canned_config_file = os.path.join(path_database, "cannedPhrasesMode.ini")
        canned_config = configparser.ConfigParser()
        file_size = len(canned_config.read(canned_config_file))
        if file_size > 0:
            canned_config_set = True
            ConvAssistLog.info("INI file for CannedPhrases mode found")
            addTextToWindow("INI file for CannedPhrases mode found \n")
            if len(path_static) > 1 and "\\" in path_static and len(path_personalized) > 1 and "\\" in path_personalized:
                canned_config.set('CannedWordPredictor', 'static_resources_path', path_static)
                canned_config.set('CannedWordPredictor', 'personalized_resources_path', path_personalized)
                canned_config.set('CannedPhrasesPredictor', 'static_resources_path', path_static)
                canned_config.set('CannedPhrasesPredictor', 'personalized_resources_path', path_personalized)
            with open(canned_config_file, 'w') as configfile:
                canned_config.write(configfile)
        else:
            ConvAssistLog.info("INI file for CannedPhrases mode NOT found")
            addTextToWindow("INI file for CannedPhrases mode NOT found \n")
    except Exception as e:
        canned_config_set = False
        ConvAssistLog.critical(f" INI file for CannedPhrases {e}")
        addTextToWindow(f"Exception INI file for CannedPhrases {e} \n")
        raise e


def deleteOldPyinstallerFolders(time_threshold=100):
    """
    Deletes the Temp folders created by Pyinstaller if they were not closed correctly
    :param time_threshold: in seconds
    :return: void
    """
    try:
        
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
            ConvAssistLog.debug(f"Directory for current _MEI Folder {base_path}")
        else:
            ConvAssistLog.info("Not running in a PyInstaller bundle.")
            return
        
        temp_path = os.path.abspath(os.path.join(base_path, '..'))  # Go to parent folder of MEIPASS
        ConvAssistLog.debug(f"temp folder path {temp_path}")

        # Search all MEIPASS folders...
        mei_folders = glob.glob(os.path.join(temp_path, '_MEI*'))
        ConvAssistLog.debug(f"_MEI folders {mei_folders}")
        count_list = len(mei_folders)
        ConvAssistLog.debug(f"_MEI Folders count {count_list}")
        for item in mei_folders:
            try:
                ConvAssistLog.debug(f"----item {item}")
                if (time.time() - os.path.getctime(item)) > time_threshold and item != base_path:
                    ConvAssistLog.debug(f"Deleting {item}")
                    if os.path.isdir(item):
                        shutil.rmtree(item)
            except Exception as es:
                ConvAssistLog.error(f" deleting folder {item}: {es}")
                raise es
    except Exception as es:
        ConvAssistLog.critical(f" Directory for _MEI Folders {es}")
        raise es


"""
Creation of the main Frame of the UI
Buttons, text and shapes
"""
licesnce_text = licesnce_text_string2
ws = Tk()
# ws.title('ConvAssist')
# frame = Frame(ws)
ws.geometry("600x350")
ws.overrideredirect(1)
ws.wm_attributes("-transparentcolor", "grey")
frame_photo = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "frame.png"))
frame_label = Label(ws, border=0, bg='grey', image=frame_photo)
frame_label.pack(fill=BOTH, expand=True)
frame_label.bind("<B1-Motion>", move_App)
ws.resizable(False, False)

label = Label(ws, text="ConvAssist", fg='#ffaa00', bg='#232433', font=("Verdana", 14))
label.place(x=60, y=12)
label2 = Label(ws, text="Messages Window", fg="black", bg='#FFFFFF', font=("Arial", 10))
label2.place(x=25, y=57.5)
label_licence = Label(ws, text=licesnce_text, fg="black", bg='#ffffff', font=("Verdana", 8))
label_licence.place(x=25, y=85, width=551, height=200)

text_box = Text(ws, font=("Arial", 12))
text_box.place(x=25, y=85, width=526, height=200)
sb = Scrollbar(ws, orient=VERTICAL)
sb.place(x=551, y=85, width=25, height=200)
text_box.config(yscrollcommand=sb.set)
sb.config(command=text_box.yview)

button_image_clear = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_clear.png"))
clear_button = Label(ws, image=button_image_clear, border=0, bg='#FFFFFF', text=" ")
clear_button.place(x=25, y=300)
clear_button.bind("<Button>", lambda e: clear_textWindow())

button_image_license = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_license.png"))
license_button = Label(ws, image=button_image_license, border=0, bg='#FFFFFF', text=" ")
license_button.place(x=150, y=300)
license_button.bind("<Button>", lambda e: show_license())

button_image_exit = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_exit.png"))
exit_button = Label(ws, image=button_image_exit, border=0, bg='#FFFFFF', text=" ")
exit_button.place(x=275, y=300)
exit_button.bind("<Button>", lambda e: hide_window())

button_image_back = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_back.png"))
back_button = Label(ws, image=button_image_back, border=0, bg='#FFFFFF', text=" ")
back_button.place(x=465, y=300)
back_button.bind("<Button>", lambda e: hide_license())
back_button.lower()

image_icon_topBar = Image.open(os.path.join(SCRIPT_DIR, "Assets", "icon_tray.png"))
resize_image = image_icon_topBar.resize((32, 23))
icon_image_ConvAssist = ImageTk.PhotoImage(resize_image)
icon_image = Label(ws, image=icon_image_ConvAssist, border=0, bg='#232433', text=" ")
icon_image.place(x=16, y=12, width=32, height=23)

"""
Start of Program
Code needed to avoid the App to Pop up windows with warnings
"""
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
"""
Config process
"""
"""
If there is more than 2 process name as ConvAssist it will not start another instance of Systray and close the most recent request of Systray App
"""
isProcessRunning = findProcessIdByName("ConvAssistCPApp.exe")
if isProcessRunning:
    sys.exit()
"""
Thread to delete old temp files and folders from .exe when they where not closed gracefully 
"""
ConvAssistLog.info("Thread to delete old temp files and folders from .exe when they where not closed gracefully")
delete_old_folders = threading.Thread(target=deleteOldPyinstallerFolders)
delete_old_folders.start()
"""
Start the thread to Set the connection with a possible active server (Loop)
"""
ConvAssistLog.info("Start the thread to Set the connection with a possible active server (Loop)")
threading.Thread(target=InitPredict).start()
"""
Define a method to be launch when the window is closed
"""
ConvAssistLog.info("Define a method to be launch when the window is closed")
ws.protocol('WM_DELETE_WINDOW', hide_window)
"""
Set the Close window after (time) mili seconds to start the SysTray Process
"""
ConvAssistLog.info("Set the Close window after (time) mili seconds to start the SysTray Process")
ws.after(10, lambda: hide_window())
"""
Start the Main window
"""
ConvAssistLog.info("Start the Main window")
ws.mainloop()
