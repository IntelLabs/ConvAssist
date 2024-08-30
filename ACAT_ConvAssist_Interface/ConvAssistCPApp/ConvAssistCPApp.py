# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
   Systray application for ConvAssist to be used with ACAT
"""

import glob
import os
from pathlib import Path
import time
import sys
import json
import threading
import sys
import psutil
import shutil
import tempfile

import PIL.Image
import pystray
from pystray import MenuItem as item

from tkinter import Tk

from configparser import ConfigParser
from typing import Any, List
from enum import IntEnum

current_path = os.path.dirname(os.path.realpath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

import ACAT_ConvAssist_Interface.ConvAssistCPApp.Win32PipeHandler as Win32PipeHandler
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ConvAssistWindow import ConvAssistWindow
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATMessageTypes import ConvAssistMessage, ConvAssistSetParam, WordAndCharacterPredictionResponse, \
    ConvAssistMessageTypes, ConvAssistPredictionTypes, ParameterType

from ConvAssist import ConvAssist
from ConvAssist.utilities.logging import ConvAssistLogger
from ConvAssist.utilities.callback import BufferedCallback

ca_main_id = "MAIN"
ca_normal_ini = "wordPredMode.ini"
ca_normal_id = "NORMAL"
ca_shorthand_ini = "shortHandMode.ini"
ca_shorthand_id = "SHORTHAND"
ca_sentence_ini = "sentenceMode.ini"
ca_sentence_id = "SENTENCE"
ca_cannedphrases_ini = "cannedPhrasesMode.ini"
ca_cannedphrases_id = "CANNED"

app_quit_event = threading.Event()

class ConvAssistVariables:
    def __init__(self):
        self.retries = 5
        self.pipeName = "ACATConvAssistPipe"

        self.logger = ConvAssistLogger(
            #Initial logger location.  updated in the setParam message
            "MAIN", "DEBUG", f"{tempfile.gettempdir()}"
        )
        self.ConvAssist_callback = BufferedCallback("")
        self.clientConnected: bool = False
        # self.convAssistCPWindow:ConvAssistWindow | None = None

        # Parameters that ACAT will send to ConvAssist
        self.path: str = ""
        self.suggestions:int = 10
        self.testgensentencepred:bool = False
        self.retrieveaac:bool = False
        self.pathstatic:str = ""
        self.pathpersonalized:str = ""
        self.enablelogs:bool = False
        self.loglevel:str = "DEBUG"
        self._pathlog = ""

        # Variables for the configuration of the predictors
        self.word_config: ConfigParser = ConfigParser()
        self.sh_config: ConfigParser = ConfigParser()
        self.sent_config: ConfigParser = ConfigParser()
        self.canned_config: ConfigParser = ConfigParser()

        self.sent_config_change: bool = False
        self.path_logs: str = ""
        self.enable_logs: bool = True

        # Variables for the predictors
        self.conv_normal: ConvAssist = ConvAssist(ca_normal_id, ca_normal_ini)
        self.conv_shorthand: ConvAssist = ConvAssist(ca_shorthand_id, ca_shorthand_ini)
        self.conv_sentence: ConvAssist = ConvAssist(ca_sentence_id, ca_sentence_ini)
        self.conv_canned_phrases: ConvAssist = ConvAssist(ca_cannedphrases_id, ca_cannedphrases_ini)

    @property
    def pathlog(self):
        return self._pathlog
    
    @pathlog.setter
    def pathlog(self, value):
        self.logger.debug(f"Setting log location to {value}.")
        self._pathlog = value
        self.logger.configure_logging(self.loglevel, self._pathlog)
    
    @pathlog.deleter
    def pathlog(self):
        del self._pathlog

def createPredictorConfig(ca_vars:ConvAssistVariables, config_file_name) -> ConfigParser | None:
    # check if all required params are set
    if not ca_vars.path or not ca_vars.pathstatic or not ca_vars.pathpersonalized:
        ca_vars.logger.debug("Path not set")
        return None
    
    # Check if config file exists
    config_file = os.path.join(ca_vars.path, config_file_name)
    if not os.path.exists(config_file):
        ca_vars.logger.debug("Config file not found")
        return None
    
    # Create a ConfigParser object and read the config file
    config_parser = ConfigParser()
    config_parser.read(config_file)

    # Check if the config file is read successfully
    if not config_parser:
        ca_vars.logger.debug("Config file not read successfully")
        return None
    
    # make additional customizations to the ConfigParser object
    for section in config_parser.sections():
        for key in config_parser[section]:
            if key == "static_resources_path":
                config_parser[section][key] = ca_vars.pathstatic
            elif key == "personalized_resources_path":
                config_parser[section][key] = ca_vars.pathpersonalized
            elif key == "suggestions":
                config_parser[section][key] = str(ca_vars.suggestions)
            elif key == "test_generalSentencePrediction":
                config_parser[section][key] = str(ca_vars.testgensentencepred)
            elif key == "retrieveAAC":
                config_parser[section][key] = str(ca_vars.retrieveaac)
            elif key == "log_location":
                config_parser[section][key] = ca_vars.pathlog
            elif key == "log_level":
                config_parser[section][key] = ca_vars.loglevel

    # Write the config file
    with open(config_file, 'w') as configfile:
        config_parser.write(configfile)

    #return the ConfigParser object
    return config_parser

def sort_List(prediction_list, amount_predictions):
    """
    Gets a specific amount of predictions in descending order

    :param prediction_list: original predictions list
    :param amount_predictions: amount of predictions requested
    :return: Sorted predictions list
    """
    new_ConvAssist_list = []
    try:
        temp_ConvAssist_list = sorted(
            prediction_list, key=lambda x: (x[1]), reverse=True
        )
        for x in range(amount_predictions):
            if x >= len(prediction_list):
                break
            element_list = temp_ConvAssist_list[x]
            new_ConvAssist_list.append(element_list)
    except Exception:
        new_ConvAssist_list = []
    return new_ConvAssist_list

def handle_incoming_messages(Pipehandle):
    """
    Main Thread, called when a server was found: Receives and send messages, Event message terminate App

    :param Pipehandle: Handle of the pipe
    :return: none
    """
    while not app_quit_event.is_set():
        ca_vars.logger.info("Handle incoming message start.")
        send_response = True

        try:
            PredictionResponse = WordAndCharacterPredictionResponse()

            try:
                message_json = Win32PipeHandler.get_incoming_message(Pipehandle)
                messageReceived = ConvAssistMessage.jsonDeserialize(message_json)
                ca_vars.logger.debug(f"Message received: {messageReceived} ")

            except TimeoutError as e:
                ca_vars.logger.debug(f"Timeout Error waiting for named pipe.  Try again.")
                continue
            
            except BrokenPipeError as e:
                # If the pipe is broken, exit the loop
                ca_vars.logger.critical(f"Broken Pipe Error. Bailing. {e}.", e)
                send_response = False
                app_quit_event.set()
                continue

            except Exception as e:
                ca_vars.logger.critical(f"Catastrophic Error.  Bailing. {e}.", e)
                # messageReceived = ConvAssistMessage(ConvAssistMessageTypes.NONE, ConvAssistPredictionTypes.NONE, "")
                send_response = False
                app_quit_event.set()
                continue

            match messageReceived.MessageType:
                case ConvAssistMessageTypes.SETPARAM:
                    config_changed = handle_parameter_change(messageReceived, ca_vars)

                    if config_changed:
                        try:
                            initialize_or_configure_convassists(ca_vars)
                        except AttributeError as e:
                            ca_vars.logger.warning(f"Initializing predictors - Config not ready: {e}.")
                            pass
                
                case ConvAssistMessageTypes.NEXTWORDPREDICTION:
                    next_word_prediction(PredictionResponse, messageReceived)

                case ConvAssistMessageTypes.NEXTSENTENCEPREDICTION:
                    next_sentence_prediction(PredictionResponse, messageReceived)
                
                case ConvAssistMessageTypes.LEARNWORDS:
                    handle_learn(ca_vars.conv_sentence, messageReceived, "WORDS")
                    PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNWORDS
                
                case ConvAssistMessageTypes.LEARNSENTENCES:
                    handle_learn(ca_vars.conv_sentence, messageReceived, "SENTENCES")
                    PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNSENTENCES

                case ConvAssistMessageTypes.LEARNCANNED:    
                    handle_learn(ca_vars.conv_canned_phrases, messageReceived, "CANNEDPHRASESMODE")
                    PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNCANNED
                
                case ConvAssistMessageTypes.LEARNSHORTHAND:
                    handle_learn(ca_vars.conv_shorthand, messageReceived, "SHORTHANDMODE")
                    PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNSHORTHAND
                                
                case ConvAssistMessageTypes.FORCEQUITAPP:
                    ca_vars.logger.info("Force Quit App message received.")
                    Win32PipeHandler.DisconnectNamedPipe(Pipehandle)
                    send_response = False
                    app_quit_event.set()

                case _:
                    send_response = False

            if send_response:
                ca_vars.logger.info(f"Sending message: {PredictionResponse}.")
                Win32PipeHandler.send_message(Pipehandle, PredictionResponse.jsonSerialize())

        except Exception as e:
            ca_vars.logger.critical(f"Critical Error in Handle incoming message. Bailing {e}.", e)
            Win32PipeHandler.DisconnectNamedPipe(Pipehandle)
            app_quit_event.set()
        
        ca_vars.logger.info("Handle incoming message finished.")

def next_word_prediction(PredictionResponse, messageReceived):
    ca_vars.logger.debug(f"Prediction requested type: {messageReceived.PredictionType} Prediction Message: {messageReceived.Data}.")
    word_prediction = []
    next_Letter_Probs = []
    sentence_nextLetterProbs = []
    sentence_predictions = []
    sentences_count = 0
    prediction_type = ConvAssistPredictionTypes.NONE

    match messageReceived.PredictionType:
        case ConvAssistPredictionTypes.NORMAL:
            prediction_type = ConvAssistPredictionTypes.NORMAL
            if ca_vars.conv_normal.initialized:
                count = len(messageReceived.Data)
                if (count == 1 and messageReceived.Data.isspace()):
                    if ca_vars.conv_normal.context_tracker.callback is not None:
                        ca_vars.conv_normal.context_tracker.callback.update("")
                else:
                    if ca_vars.conv_normal.context_tracker.callback is not None:
                        ca_vars.conv_normal.context_tracker.callback.update(messageReceived.Data)
                    
                ca_vars.conv_normal.context_tracker.prefix()
                ca_vars.conv_normal.context_tracker.past_stream()

                (next_Letter_Probs,
                word_prediction,
                sentence_nextLetterProbs,
                sentence_predictions) = ca_vars.conv_normal.predict()

        case ConvAssistPredictionTypes.SHORTHANDMODE:
            prediction_type = ConvAssistPredictionTypes.SHORTHANDMODE
            if ca_vars.conv_shorthand.initialized:
                sentences_count = 0
                if ca_vars.conv_shorthand.context_tracker.callback is not None:
                    ca_vars.conv_shorthand.context_tracker.callback.update(messageReceived.Data)

                ca_vars.conv_shorthand.context_tracker.prefix()
                ca_vars.conv_shorthand.context_tracker.past_stream()

                (next_Letter_Probs,
                word_prediction,
                sentence_nextLetterProbs,
                sentence_predictions) = ca_vars.conv_shorthand.predict()

        case ConvAssistPredictionTypes.CANNEDPHRASESMODE:
            prediction_type = ConvAssistPredictionTypes.CANNEDPHRASESMODE
            if ca_vars.conv_canned_phrases.initialized:
                sentences_count = 6
                if ca_vars.conv_canned_phrases.context_tracker.callback is not None:
                    ca_vars.conv_canned_phrases.context_tracker.callback.update(messageReceived.Data)

                ca_vars.conv_canned_phrases.context_tracker.prefix()
                ca_vars.conv_canned_phrases.context_tracker.past_stream()

                (next_Letter_Probs,
                 word_prediction,
                 sentence_nextLetterProbs,
                 sentence_predictions) = ca_vars.conv_canned_phrases.predict()

    next_Letter_Probs = sort_List(next_Letter_Probs, 20)
    word_prediction = sort_List(word_prediction, 10)
    sentence_nextLetterProbs = sort_List(sentence_nextLetterProbs, 0)
    sentence_predictions = sort_List(sentence_predictions, sentences_count)

    #TODO - Check if this should be json.dumps instead?
    result_Letters = str(next_Letter_Probs)
    result_Words = str(word_prediction)
    result_Letters_Sentence = str(sentence_nextLetterProbs)
    result_Sentences = str(sentence_predictions)

    PredictionResponse.MessageType = ConvAssistMessageTypes.NEXTWORDPREDICTIONRESPONSE
    PredictionResponse.PredictionType = prediction_type
    PredictionResponse.PredictedWords = result_Words
    PredictionResponse.NextCharacters = result_Letters
    PredictionResponse.NextCharactersSentence = result_Letters_Sentence
    PredictionResponse.PredictedSentence = result_Sentences

def next_sentence_prediction(PredictionResponse, messageReceived):
    ca_vars.logger.debug(f"Prediction requested type: {messageReceived.MessageType} Prediction Message: {messageReceived.Data}.")
    word_prediction = []
    next_Letter_Probs = []
    sentence_nextLetterProbs = []
    sentence_predictions = []
    sentences_count = 0
    prediction_type = ConvAssistPredictionTypes.SENTENCES
    
    if ca_vars.conv_sentence.initialized and ca_vars.conv_sentence.check_model() == 1:
        if ca_vars.conv_sentence.context_tracker.callback is not None:
            ca_vars.conv_sentence.context_tracker.callback.update(messageReceived.Data)
        ca_vars.conv_sentence.context_tracker.prefix()
        ca_vars.conv_sentence.context_tracker.past_stream()
        next_Letter_Probs, word_prediction, sentence_nextLetterProbs, sentence_predictions = ca_vars.conv_sentence.predict()

    next_Letter_Probs = sort_List(next_Letter_Probs, 0)
    word_prediction = sort_List(word_prediction, 0)
    sentence_nextLetterProbs = sort_List(sentence_nextLetterProbs, 0)
    sentence_predictions = sort_List(sentence_predictions, 6)
    result_Letters = str(next_Letter_Probs)
    result_Words = str(word_prediction)
    result_Letters_Sentence = str(sentence_nextLetterProbs)
    result_Sentences = str(sentence_predictions)

    PredictionResponse.MessageType = ConvAssistMessageTypes.NEXTSENTENCEPREDICTIONRESPONSE
    PredictionResponse.PredictionType = prediction_type
    PredictionResponse.PredictedWords = result_Words
    PredictionResponse.NextCharacters = result_Letters
    PredictionResponse.NextCharactersSentence = result_Letters_Sentence
    PredictionResponse.PredictedSentence = result_Sentences

def handle_learn(conv_assist: ConvAssist, messageReceived: ConvAssistMessage, mode: str):
    ca_vars.logger.debug(f"Calling Learn_db for {conv_assist.id} with mode {mode}.")
    conv_assist.learn_db(messageReceived.Data)
    ca_vars.logger.debug("Finished Learn_db for {conv_assist.id}.")

def handle_parameter_change(messageReceived: ConvAssistMessage, ca_vars: ConvAssistVariables):
    changed = False

    try:
        param = ConvAssistSetParam.jsonDeserialize(messageReceived.Data)
        ca_vars.logger.debug(f"Parameter change requested: {param}.")

        attr_name = ParameterType(param.Parameter).name.lower()
        ca_vars.logger.debug(f"Parameter {attr_name} with value {param.Value}.")
        old_value = getattr(ca_vars, attr_name, None)

        if old_value != param.Value:
            changed = True
            setattr(ca_vars, attr_name, param.Value)

    except Exception as e:
        ca_vars.logger.error(f"handle_parameters exception: {e}.")
        
    ca_vars.logger.info("Parameters message answered.")
    return changed

def initialize_or_configure_convassists(ca_vars: ConvAssistVariables):
    convassists = [ca_vars.conv_normal, ca_vars.conv_shorthand, ca_vars.conv_sentence, ca_vars.conv_canned_phrases]

    for convassist in convassists:
        if convassist.initialized:
            convassist.update_params(str(ca_vars.testgensentencepred), str(ca_vars.retrieveaac))
            convassist.read_updated_toxicWords()
            ca_vars.logger.info(f"convassist {convassist.id} updated.")
        else:
            try:
                config = createPredictorConfig(ca_vars, f"{convassist.ini_file}")
                if config:
                    convassist.initialize(config, ca_vars.ConvAssist_callback, ca_vars.pathlog, ca_vars.loglevel)
                
                    if convassist.id == ca_sentence_id:
                        convassist.read_updated_toxicWords()
                    if convassist.id == ca_cannedphrases_id:
                        ca_vars.conv_canned_phrases.cannedPhrase_recreateDB()

                    ca_vars.logger.info(f"convassist {convassist.id} initialized and configured.")

            except Exception as e:
                ca_vars.logger.critical(f"Error initializing convassist {convassist.id}: {e}.")
                raise e
    
def ConnectToACAT() -> tuple[bool, Any]:

    success = False
    handle = None
    # Try to connect to ACAT server.  Give up after #retries
    ca_vars.logger.info("Trying to connect to ACAT server.")
    try:
        success, handle = Win32PipeHandler.ConnectToNamedPipe(ca_vars.pipeName, ca_vars.retries, ca_vars.logger)

        if not success:
            ca_vars.logger.info("Failed to connect to ACAT server.")

        else:
            ca_vars.logger.info("Connected to ACAT server.")
            ca_vars.clientConnected = True

    except Exception as e:
        ca_vars.logger.error(f"Error connecting to named pipe: {e}")

    finally:
        return success, handle

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
            pinfo = proc.as_dict(attrs=["pid", "name", "create_time"])
            # Check if process name contains the given name string.
            if process_name.lower() in pinfo["name"].lower():
                listOfProcessObjects.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    ca_vars.logger.debug(
        f"Number of processes with name ConvAssist.exe: {len(listOfProcessObjects)}"
    )
    if len(listOfProcessObjects) > 2:
        return True
    return False

def deleteOldPyinstallerFolders(time_threshold=100):
    """
    Deletes the Temp folders created by Pyinstaller if they were not closed correctly
    :param time_threshold: in seconds
    :return: void
    """
    logger = ca_vars.logger
    try:
        if not getattr(sys, "frozen", False):
            logger.info("Not running in a PyInstaller bundle.")
            return

        temp_path = tempfile.gettempdir()
        logger.debug(f"temp folder path {temp_path}")

        # Search all MEIPASS folders...
        mei_folders = glob.glob(os.path.join(temp_path, "_MEI*"))
        logger.debug(f"_MEI folders {mei_folders}")
        count_list = len(mei_folders)
        logger.debug(f"_MEI Folders count {count_list}")
        for item in mei_folders:
            try:
                logger.debug(f"Checking {item}")
                if (
                    time.time() - os.path.getctime(item)
                ) > time_threshold:  # and item != base_path:
                    logger.debug(f"Deleting {item}")
                    if os.path.isdir(item):
                        pass
                        shutil.rmtree(item)
            except Exception as es:
                logger.error(f"Deleting folder {item}: {es}")
                raise es
    except Exception as es:
        logger.critical(f"Iterating directory for _MEI Folders: {es}")
        raise es

"""
Start of Program
Code needed to avoid the App to Pop up windows with warnings
"""
if not sys.warnoptions:
    import warnings

warnings.simplefilter("ignore")

# def show_main_window():
#     if not ca_vars.convAssistCPWindow:
#         ca_vars.convAssistCPWindow = ConvAssistWindow(ca_vars.logger)
#         ca_vars.convAssistCPWindow.mainloop()
#     else:
#         ca_vars.convAssistCPWindow.show_action()

# def show_window(icon, window = None):
#     icon.stop()
#     if window: window.deiconify()

# def hide_window(icon, window = None):
#     if window: window.withdraw()
#     icon.run()

# def quit_application(icon, window:ConvAssistWindow):
#     app_quit_event.set()
#     if icon:
#         icon.stop()
#     if window:
#         # window.quit
#         window.destroy()

# def setup_window():
#     window = ConvAssistWindow(ca_vars.logger)
#     window.protocol("WM_DELETE_WINDOW", lambda: hide_window(convAssistCPIcon, convAssistCPWindow))

#     return window

# def setup_tray_icon(window = None):
#     # SysTray Icon
#     menu = (item("More Info...", lambda: show_window(icon, window)), item('Exit', lambda: quit_application(convAssistCPIcon, convAssistCPWindow)))
#     image = PIL.Image.open(os.path.join(current_path, "Assets", "icon_tray.png"))
#     icon = pystray.Icon("test_icon", image, "ConvAssist", menu)
#     return icon

ca_vars: ConvAssistVariables = ConvAssistVariables()

def main():

    if findProcessIdByName("ConvAssist.exe"):
        sys.exit()

    success, handle = ConnectToACAT()
    if not success:
        sys.exit()

    message_thread = threading.Thread(target=handle_incoming_messages, args=(handle,))
    message_thread.start()

# convAssistCPWindow = setup_window()
# convAssistCPIcon = setup_tray_icon(None)


# icon_thread = threading.Thread(target=convAssistCPIcon.run)
# # icon_thread.daemon = True
# icon_thread.start()

# convAssistCPWindow.mainloop()

# icon_thread.join()

# if icon_thread.is_alive():
#     convAssistCPIcon.stop()
#     icon_thread.join()

if __name__ == "__main__":
    deleteOldPyinstallerFolders()
    main()
    app_quit_event.wait()
    sys.exit()