# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser

from nltk import sent_tokenize

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictior_activator import PredictorActivator
from ConvAssist.predictor_registry import PredictorRegistry
from ConvAssist.utilities.logging_utility import LoggingUtility


class ConvAssist:
    """
    ConvAssist class represents a conversational assistant.

    Methods:
    - __init__(self, id: str, ini_file: str, config: ConfigParser | None = None, log_location: str | None = None, log_level: int | None = None)
        Initializes the ConvAssist object with the provided parameters.
    - initialize(self, config: ConfigParser, log_location: str | None = None, log_level: int | None = None)
        Initializes the ConvAssist object with the provided configuration, log location, and log level.
    - predict(self)
        Calls the predict function on the predictor_activator to make language model predictions.
    - update_params(self, test_gen_sentence_pred, retrieve_from_AAC)
        Updates the parameters in ACAT and syncs them with the initialized parameters in ConvAssist Predictors.
    - read_updated_toxicWords(self)
        Reads the updated toxic words from the predictor_activator.
    - setLogLocation(self, filename, pathLoc, level)
        Sets the log location for ConvAssist's logger.
    - cannedPhrase_recreateDB(self)
        Recreates the databases for the cannedPhrases predictor.
    - learn_db(self, text)
        Learns a sentence, word, or phrase.
    - check_model(self)
        Checks if models associated with a predictor are loaded.
    """

    def __init__(
        self,
        name: str,
        ini_file: str | None = None,
        config: ConfigParser | None = None,
        log_location: str | None = None,
        log_level: int = logging.ERROR,
    ):
        """
        Initializes an instance of the class.
        Args:
            id (str): The identifier of the instance.
            ini_file (str): The path to the ini file.
            config (ConfigParser | None, optional): The configuration parser object. Defaults to None.
            log_location (str | None, optional): The location to store log files. Defaults to None.
            log_level (int | None, optional): The log level. Defaults to None.
        """
        self.config = config
        self.log_level = log_level
        self.log_location = log_location
        self.initialized = False
        self.name = name
        self.ini_file = ini_file

        if self.config:
            self.initialize(self.config, self.log_location, self.log_level)

    def initialize(
        self, config: ConfigParser, log_location: str | None = None, log_level: int | None = None
    ):
        """
        Initializes the ConvAssist object with the provided configuration, log location, and log level.
        Args:
            config (ConfigParser): The configuration parser object.
            log_location (str | None, optional): The location to store log files. Defaults to None.
            log_level (int | None, optional): The log level. Defaults to None.
        """
        if not config:
            raise AttributeError("Config not provided.")

        if not self.config:
            self.config = config

        if log_level:
            self.log_level = log_level

        if log_location:
            self.log_location = log_location

        self.logger = LoggingUtility().get_logger(
            self.name, self.log_level, self.log_location, True
        )

        lowercase_mode = self.config.getboolean("ContextTracker", "lowercase_mode", fallback=False)
        self.context_tracker = ContextTracker(lowercase_mode)

        self.predictor_registry = PredictorRegistry()
        self.predictor_registry.set_predictors(self.config, self.context_tracker, self.logger)

        self.predictor_activator = PredictorActivator(
            self.config, self.predictor_registry, self.context_tracker, self.logger
        )
        self.predictor_activator.combination_policy = "meritocracy"

        self.initialized = True

    def predict(self) -> tuple:
        """
        Calls the predict function on the predictor_activator to make language model predictions.
        Returns:
            tuple: The predictions made by the predictor_activator.
        """
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        multiplier = 1
        (wordprob, word, sentprob, sent) = self.predictor_activator.predict(multiplier)
        if word != []:
            # normalize word probabilities over 10 words.
            normalized_words = word[0:10]
            prob_sum_over10 = 0.0
            words = []
            for w in enumerate(normalized_words):
                prob_sum_over10 += w[1].probability
                words.append(w[1].word)

        return (
            wordprob,
            [(p.word, p.probability / prob_sum_over10) for p in word],
            sentprob,
            [(p.word, p.probability) for p in sent],
        )

    def update_params(self, test_gen_sentence_pred, retrieve_from_AAC):
        self.predictor_activator.update_params(test_gen_sentence_pred, retrieve_from_AAC)

    def read_updated_toxicWords(self):
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        self.predictor_activator.read_updated_toxicWords()

    # def setLogLocation(self, filename, pathLoc , level):
    #     # self.predictor_activator.set_log(filename, pathLoc, level)
    #     pass

    def recreate_database(self):
        """
        Recreates the databases for the cannedPhrases predictor.
        """
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        self.predictor_activator.recreate_database()

    def learn_text(self, text):
        """
        Learns a sentence, word, or phrase.
        Args:
            text (str): The text to learn.
        """
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        sentences = sent_tokenize(text)
        for eachSent in sentences:
            self.predictor_activator.learn_text(eachSent)

    def check_model(self):
        """
        Checks if models associated with a predictor are loaded.
        Returns:
            Any: The status of the model.
        """
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        status = self.predictor_registry.model_status()
        return status
