# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

"""
   Systray application for ConvAssist to be used with ACAT
"""

import logging
import os
import sys
import threading
from configparser import ConfigParser
from typing import Any

from ..utilities.ACATMessageTypes import (
    ConvAssistMessage,
    ConvAssistMessageTypes,
    ConvAssistPredictionTypes,
    ConvAssistSetParam,
    ParameterType,
    WordAndCharacterPredictionResponse,
)
from interfaces.ACAT.utilities.message_handler.MessageHandler import MessageHandler

from convassist.ConvAssist import ConvAssist
from convassist.utilities.logging_utility import LoggingUtility

current_path = os.path.dirname(os.path.realpath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

ca_main_id = "ACATINTERFACE"

ca_normal_ini = "wordPredMode.ini"
ca_normal_id = "WORDPREDICTOR"

ca_shorthand_ini = "shortHandMode.ini"
ca_shorthand_id = "SHORTHANDPREDICTOR"

ca_sentence_ini = "sentenceMode.ini"
ca_sentence_id = "SENTENCEPREDICTOR"

ca_cannedphrases_ini = "cannedPhrasesMode.ini"
ca_cannedphrases_id = "CANNEDPREDICTOR"

log_level = logging.DEBUG


class ACATConvAssistInterface(threading.Thread):
    """Class to handle the ConvAssist Interface with ACAT"""

    def __init__(
        self,
        app_quit_event: threading.Event,
        queue_handler: bool = False,
        logging_level: int = log_level,
    ):
        super().__init__()

        self.logger = LoggingUtility().get_logger(
            ca_main_id, logging_level, queue_handler=queue_handler
        )

        self.app_quit_event = app_quit_event
        self.daemon = True

        self.retries = 5
        self.pipeName = "ACATConvAssistPipe"

        self.clientConnected: bool = False

        # Message Handler
        self.messageHandler = MessageHandler.getMessageHandler(
            {"type": "win32", "pipe_name": self.pipeName}
        )

        # Parameters that ACAT will send to ConvAssist
        self.path: str = ""
        self.suggestions: int = 10
        self.testgensentencepred: bool = False
        self.retrieveaac: bool = False
        self.pathstatic: str = ""
        self.pathpersonalized: str = ""
        self.enablelogs: bool = False
        self.loglevel: int = logging.DEBUG
        self._pathlog = ""

        # Variables for the configuration of the instances of ConvAssist
        self.word_config: ConfigParser = ConfigParser()
        self.sh_config: ConfigParser = ConfigParser()
        self.sent_config: ConfigParser = ConfigParser()
        self.canned_config: ConfigParser = ConfigParser()

        self.sent_config_change: bool = False
        self.enable_logs: bool = True

        self.convAssists = {}

        # instances of ConvAssist
        self.conv_normal: ConvAssist = ConvAssist(ca_normal_id, ca_normal_ini)
        self.convAssists.update({ConvAssistPredictionTypes.NORMAL: self.conv_normal})

        self.conv_shorthand: ConvAssist = ConvAssist(ca_shorthand_id, ca_shorthand_ini)
        self.convAssists.update({ConvAssistPredictionTypes.SHORTHANDMODE: self.conv_shorthand})

        self.conv_sentence: ConvAssist = ConvAssist(ca_sentence_id, ca_sentence_ini)
        self.convAssists.update({ConvAssistPredictionTypes.SENTENCES: self.conv_sentence})

        self.conv_canned_phrases: ConvAssist = ConvAssist(
            ca_cannedphrases_id, ca_cannedphrases_ini
        )
        self.convAssists.update(
            {ConvAssistPredictionTypes.CANNEDPHRASESMODE: self.conv_canned_phrases}
        )

    @property
    def pathlog(self):
        return self._pathlog

    @pathlog.setter
    def pathlog(self, value):
        self.logger.debug(f"Setting log location to {value}.")
        self._pathlog = value
        # LoggingUtility.add_file_handler(self.logger, self._pathlog )

    @pathlog.deleter
    def pathlog(self):
        del self._pathlog

    @staticmethod
    def sort_List(prediction_list, amount_predictions):
        """
        Gets a specific amount of predictions in descending order

        :param prediction_list: original predictions list
        :param amount_predictions: amount of predictions requested
        :return: Sorted predictions list
        """
        new_ConvAssist_list = []
        try:
            temp_ConvAssist_list = sorted(prediction_list, key=lambda x: (x[1]), reverse=True)
            for x in range(amount_predictions):
                if x >= len(prediction_list):
                    break
                element_list = temp_ConvAssist_list[x]
                new_ConvAssist_list.append(element_list)
        except Exception:
            new_ConvAssist_list = []
        return new_ConvAssist_list

    def createPredictorConfig(self, config_file_name) -> ConfigParser | None:
        # check if all required params are set
        if not self.path or not self.pathstatic or not self.pathpersonalized:
            self.logger.debug("Path not set")
            return None

        # Check if config file exists
        config_file = os.path.join(self.path, config_file_name)
        if not os.path.exists(config_file):
            self.logger.debug("Config file not found")
            return None

        # Create a ConfigParser object and read the config file
        config_parser = ConfigParser()
        config_parser.read(config_file)

        # Check if the config file is read successfully
        if not config_parser:
            self.logger.debug("Config file not read successfully")
            return None

        # make additional customizations to the ConfigParser object
        for section in config_parser.sections():
            for key in config_parser[section]:
                if key == "static_resources_path":
                    config_parser[section][key] = self.pathstatic
                elif key == "personalized_resources_path":
                    config_parser[section][key] = self.pathpersonalized
                elif key == "suggestions":
                    config_parser[section][key] = str(self.suggestions)
                elif key == "_test_generalsentenceprediction":
                    config_parser[section][key] = str(self.testgensentencepred)
                elif key == "retrieveAAC":
                    config_parser[section][key] = str(self.retrieveaac)
                elif key == "log_location":
                    config_parser[section][key] = self.pathlog
                elif key == "log_level":
                    # config_parser[section][key] = logging.getLevelName(self.loglevel)
                    config_parser[section][key] = logging.getLevelName(log_level)

        # Write the config file
        with open(config_file, "w") as configfile:
            config_parser.write(configfile)

        # return the ConfigParser object
        return config_parser

    def handle_incoming_messages(self):
        """
        Main Thread, called when a server was found: Receives and send messages, Event message terminate App

        :param Pipehandle: Handle of the pipe
        :return: none
        """
        while not self.app_quit_event.is_set():
            self.logger.info("Handle incoming message start.")
            send_response = True

            try:
                PredictionResponse = WordAndCharacterPredictionResponse()

                try:
                    message_json = self.messageHandler.receive_message()
                    messageReceived = ConvAssistMessage.jsonDeserialize(message_json)
                    self.logger.info(f"Message received: {messageReceived} ")

                except TimeoutError as e:
                    self.logger.info("Timeout Error waiting for named pipe.  Try again.")
                    continue

                except BrokenPipeError as e:
                    # If the pipe is broken, exit the loop
                    self.logger.critical(f"Broken Pipe Error. Bailing. {e}.")
                    send_response = False
                    self.app_quit_event.set()
                    continue

                except Exception as e:
                    self.logger.critical(f"Catastrophic Error.  Bailing. {e}.")
                    send_response = False
                    self.app_quit_event.set()
                    continue

                match messageReceived.MessageType:
                    case ConvAssistMessageTypes.SETPARAM:
                        config_changed = self.handle_parameter_change(messageReceived)

                        if config_changed:
                            try:
                                self.initialize_or_configure_convassists()
                            except AttributeError as e:
                                self.logger.warning(f"ConfigAssist not ready: {e}.")
                                pass

                    case ConvAssistMessageTypes.NEXTWORDPREDICTION:
                        self.next_word_prediction(PredictionResponse, messageReceived)

                    case ConvAssistMessageTypes.NEXTSENTENCEPREDICTION:
                        self.next_sentence_prediction(PredictionResponse, messageReceived)

                    case ConvAssistMessageTypes.LEARNWORDS:
                        self.handle_learn(self.conv_sentence, messageReceived, "WORDS")
                        PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNWORDS

                    case ConvAssistMessageTypes.LEARNSENTENCES:
                        self.handle_learn(self.conv_sentence, messageReceived, "SENTENCES")
                        PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNSENTENCES

                    case ConvAssistMessageTypes.LEARNCANNED:
                        self.handle_learn(
                            self.conv_canned_phrases, messageReceived, "CANNEDPHRASESMODE"
                        )
                        PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNCANNED

                    case ConvAssistMessageTypes.LEARNSHORTHAND:
                        self.handle_learn(self.conv_shorthand, messageReceived, "SHORTHANDMODE")
                        PredictionResponse.MessageType = ConvAssistMessageTypes.LEARNSHORTHAND

                    case ConvAssistMessageTypes.FORCEQUITAPP:
                        self.logger.info("Force Quit App message received.")
                        self.messageHandler.disconnect()
                        send_response = False
                        self.app_quit_event.set()

                    case _:
                        send_response = False

                if send_response:
                    self.logger.info(f"Sending message: {PredictionResponse}.")
                    self.messageHandler.send_message(PredictionResponse.jsonSerialize())

            # TODO - Handle more gracefully
            except Exception as e:
                self.logger.critical(f"Critical Error in Handle incoming message. Bailing {e}.")
                self.messageHandler.disconnect()
                self.app_quit_event.set()

            self.logger.info("Handle incoming message finished.")

    def next_word_prediction(self, PredictionResponse, messageReceived):
        words_count = 10
        next_word_letter_count = 20
        sentences_count = 6
        next_sentence_letter_count = 6

        PredictionResponse.MessageType = ConvAssistMessageTypes.NEXTWORDPREDICTIONRESPONSE

        self.make_prediction(
            messageReceived,
            PredictionResponse,
            words_count,
            next_word_letter_count,
            sentences_count,
            next_sentence_letter_count,
        )

    def next_sentence_prediction(self, PredictionResponse, messageReceived):
        words_count = 0
        next_word_letter_count = 0
        sentences_count = 6
        next_sentence_letter_count = 6

        PredictionResponse.MessageType = ConvAssistMessageTypes.NEXTSENTENCEPREDICTIONRESPONSE

        # TODO: Due to a bug in ACAT, the prediction type is not being set correctly.  This is a workaround.
        messageReceived.PredictionType = ConvAssistPredictionTypes.SENTENCES

        self.make_prediction(
            messageReceived,
            PredictionResponse,
            words_count,
            next_word_letter_count,
            sentences_count,
            next_sentence_letter_count,
        )

    def make_prediction(
        self,
        messageReceived,
        PredictionResponse,
        words_count,
        next_word_letter_count,
        sentences_count,
        next_sentence_letter_count,
    ):
        word_prediction = []
        next_Letter_Probs = []
        sentence_nextLetterProbs = []
        sentence_predictions = []

        prediction_type = messageReceived.PredictionType
        convAssistInstance: ConvAssist = self.convAssists.get(messageReceived.PredictionType, None)

        if (
            convAssistInstance
            and convAssistInstance.initialized
            and convAssistInstance.context_tracker
        ):
            convAssistInstance.context_tracker.context = messageReceived.Data

            # Don't send any sentence predictions from NORMAL mode
            if messageReceived.PredictionType == ConvAssistPredictionTypes.NORMAL:
                sentences_count = 0
                next_sentence_letter_count = 0

            (
                next_Letter_Probs,
                word_prediction,
                sentence_nextLetterProbs,
                sentence_predictions,
            ) = convAssistInstance.predict()

        else:
            self.logger.warning(
                f"ConvAssist instance not found or not initialized for {messageReceived.PredictionType}."
            )

        next_Letter_Probs = ACATConvAssistInterface.sort_List(
            next_Letter_Probs, next_word_letter_count
        )
        word_prediction = ACATConvAssistInterface.sort_List(word_prediction, words_count)
        sentence_nextLetterProbs = ACATConvAssistInterface.sort_List(
            sentence_nextLetterProbs, next_sentence_letter_count
        )
        sentence_predictions = ACATConvAssistInterface.sort_List(
            sentence_predictions, sentences_count
        )

        result_Letters = str(next_Letter_Probs)
        result_Words = str(word_prediction)
        result_Letters_Sentence = str(sentence_nextLetterProbs)
        result_Sentences = str(sentence_predictions)

        PredictionResponse.PredictionType = prediction_type
        PredictionResponse.PredictedWords = result_Words
        PredictionResponse.NextCharacters = result_Letters
        PredictionResponse.NextCharactersSentence = result_Letters_Sentence
        PredictionResponse.PredictedSentence = result_Sentences

    def handle_learn(self, conv_assist: ConvAssist, messageReceived: ConvAssistMessage, mode: str):
        self.logger.debug(f"Calling Learn_db for {conv_assist.name} with mode {mode}.")
        conv_assist.learn_text(messageReceived.Data)
        self.logger.debug(f"Finished Learn_db for {conv_assist.name}.")

    def handle_parameter_change(self, messageReceived: ConvAssistMessage):
        changed = False

        try:
            param = ConvAssistSetParam.jsonDeserialize(messageReceived.Data)
            self.logger.debug(f"Parameter change requested: {param}.")

            attr_name = ParameterType(param.Parameter).name.lower()
            self.logger.debug(f"Parameter {attr_name} with value {param.Value}.")
            old_value = getattr(self, attr_name, None)

            if old_value != param.Value:
                changed = True
                setattr(self, attr_name, param.Value)

        except Exception as e:
            self.logger.error(f"handle_parameters exception: {e}.")

        self.logger.info("Parameters message answered.")
        return changed

    def initialize_or_configure_convassists(self):
        convassists = [
            self.conv_normal,
            self.conv_shorthand,
            self.conv_sentence,
            self.conv_canned_phrases,
        ]

        for convassist in convassists:
            if not convassist.initialized:
                try:
                    config = self.createPredictorConfig(f"{convassist.ini_file}")
                    if config:
                        convassist.initialize(config, self.pathlog, self.loglevel)
                        self.logger.info(f"convassist {convassist.name} initialized.")

                except Exception as e:
                    self.logger.critical(f"Error initializing convassist {convassist.name}: {e}.")
                    raise e

            convassist.recreate_database()
            convassist.update_params(str(self.testgensentencepred), str(self.retrieveaac))
            convassist.read_updated_toxicWords()

            self.logger.info(f"convassist {convassist.name} updated.")

    def ConnectToACAT(self, connection_type=None) -> bool:

        # success = False
        # handle = None
        # Try to connect to ACAT server.  Give up after #retries
        self.logger.info("Trying to connect to ACAT server.")
        try:
            connected = self.messageHandler.connect()

            if connected:
                self.logger.info("Connected to ACAT server.")
                self.clientConnected = True

        except Exception as e:
            self.logger.error(f"Error connecting to named pipe: {e}")

        finally:
            return self.clientConnected

    def DisconnectFromACAT(self):
        if self.clientConnected:
            self.messageHandler.disconnect()
            self.clientConnected = False

    def run(self):
        """
        Main function to start the application
        """
        self.logger.info("Starting ACATConvAssistInterface.")

        if not self.ConnectToACAT():
            self.logger.info("Failed to connect to ACAT server. Exiting.")
            self.app_quit_event.set()
            return

        # self.initialize_or_configure_convassists()
        self.handle_incoming_messages()

        self.logger.info("Disconnecting from ACAT.")
        self.DisconnectFromACAT()

        self.logger.info("ACATConvAssistInterface finished.")
