# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser

import nltk

from convassist.context_tracker import ContextTracker
from convassist.predictor_activator import PredictorActivator
from convassist.predictor_registry import PredictorRegistry
from convassist.utilities.logging_utility import LoggingUtility


class ConvAssist:
    """
    ConvAssist class represents a conversational assistant.
    """

    def __init__(
        self,
        name: str,
        ini_file: str | None = None,
        config: ConfigParser | None = None,
        log_file: bool = True,
        log_level: int = logging.DEBUG,
    ):
        self.config = config
        self.log_level = log_level
        self.log_file = log_file
        self.initialized = False
        self.name = name
        self.ini_file = ini_file

        if self.config:
            self.initialize(self.config, self.log_file, self.log_level)

    def initialize(
        self, config: ConfigParser, log_file: bool, log_level: int | None = None
    ):
        if not config:
            raise AttributeError("Config not provided.")

        if not self.config:
            self.config = config

        if log_level:
            self.log_level = log_level

        self.logger = LoggingUtility().get_logger(
            self.name, self.log_level, self.log_file, True
        )

        # Verify that the nltk files are downloaded
        self._verify_nltk_files()

        lowercase_mode = self.config.getboolean("ContextTracker", "lowercase_mode", fallback=False)
        self.context_tracker = ContextTracker(lowercase_mode)

        self.predictor_registry = PredictorRegistry()
        self.set_predictors()

        self.predictor_activator = PredictorActivator(
            self.config, self.predictor_registry, self.context_tracker, self.logger
        )
        self.predictor_activator.combination_policy = "meritocracy"

        self.initialized = True

    def _verify_nltk_files(self):
        try:
            # Check if punkt is already downloaded
            nltk.data.find("tokenizers/punkt")
            self.logger.debug("Punkt Tokenizer Models are already installed.")
        except LookupError:
            # If not, download punkt tokenizer
            self.logger.debug("Punkt Tokenizer Models not found, downloading...")
            nltk.download(["punkt", "punkt_tab"])
            self.logger.debug("Punkt Tokenizer Models downloaded successfully.")

    def predict(self) -> tuple:
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        word_nextLetterProbs = []
        word_nextWords = []
        keywords = []
        sentence_nextLetterProbs = []
        sentence_nextSentences = []
        keyword_responses = []

        multiplier = 1
        (
            word_nextLetterProbs,
            word_nextWords,
            sentence_nextLetterProbs,
            sentence_nextSentences,
        ) = self.predictor_activator.predict(multiplier)
        if word_nextWords != []:
            # normalize word probabilities over 10 words.
            prob_sum_over10 = 0.0
            for w in enumerate(word_nextWords[0:10]):
                prob_sum_over10 += w[1].probability

        return (
            word_nextLetterProbs,
            [(p.word, p.probability / prob_sum_over10) for p in word_nextWords],
            sentence_nextLetterProbs,
            [(p.word, p.probability) for p in sentence_nextSentences],
        )

    def update_params(self, test_gen_sentence_pred, retrieve_from_AAC):
        self.predictor_activator.update_params(test_gen_sentence_pred, retrieve_from_AAC)

    def read_updated_toxicWords(self):
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        self.predictor_activator.read_updated_toxicWords()

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

        sentences = nltk.sent_tokenize(text)
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

    def set_log_level(self, log_level):
        self.logger.setLevel(log_level)

    def list_predictors(self):
        """
        Lists the predictors.
        """
        if not self.initialized:
            raise AttributeError(f"ConvAssist {self.name} not initialized.")

        return self.predictor_registry.list_predictors()

    def set_predictors(self, predictors: list | None = None):
        """
        Sets the predictors.
        Args:
            predictors (list): The list of predictors.
        """
        self.predictor_registry.set_predictors(
            self.config, self.context_tracker, self.logger, predictors
        )
