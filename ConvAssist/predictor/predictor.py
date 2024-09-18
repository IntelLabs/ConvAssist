# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
from abc import ABC, abstractmethod
from configparser import ConfigParser

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.utilities.logging_utility import LoggingUtility


class Predictor(ABC):
    """
    Abstract class for predictors
    Methods:
    - __init__(self, config: ConfigParser, context_tracker: ContextTracker, predictor_name: str, short_desc: str | None = None, long_desc: str | None = None, logger: logging.Logger | None = None)
        Initializes the Predictor object with the provided parameters.
    - predict(self, max_partial_prediction_size = None, filter = None) -> tuple[Prediction, Prediction]
        Predicts the next word and sentence based on the context.
    - learn(self, change_tokens = None) -> None
        Learns from the context.
    - _read_config(self) -> None
        Reads the configuration file.
    - load_model(*args, **kwargs) -> None
        Loads the model.
    - read_personalized_toxic_words(self, *args, **kwargs) -> None
        Reads the personalized toxic words.
    """

    def __init__(
        self,
        config: ConfigParser,
        context_tracker: ContextTracker,
        predictor_name: str,
        logger: logging.Logger | None = None,
    ):

        self.config: ConfigParser = config
        self.context_tracker: ContextTracker = context_tracker
        self._predictor_name: str = predictor_name

        self._aac_dataset: str = ""  # Path
        self._blacklist_file: str = ""  # Path
        self._database: str = ""  # Path
        self._deltas: str = "0.01 0.1 0.89"
        self._embedding_cache_path: str = ""  # Path
        self._generic_phrases: str = ""  # Path
        self._index_path: str = ""  # Path
        self._learn: bool = False
        self._modelname: str = ""  # Path
        self._personalized_allowed_toxicwords_file: str = ""  # Path
        self._personalized_cannedphrases: str = ""  # Path
        self._personalized_resources_path: str = ""
        self._predictor_class: str = ""
        self._retrieve_database: str = ""  # Path
        self._retrieveaac: bool = True
        self._sbertmodel: str = ""
        self._sent_database: str = ""  # Path
        self._sentence_transformer_model: str = ""  # Path
        self._sentences_db: str = ""  # Path
        self._spellingdatabase: str = ""  # Path
        self._startsents: str = ""  # Path
        self._startwords: str = ""  # Path
        self._static_resources_path: str = ""
        self._stopwords: str = ""  # Path
        self._test_generalsentenceprediction: bool = False
        self._tokenizer: str = ""  # Path

        # configure a logger
        if logger is None:
            self.logger = LoggingUtility().get_logger(
                f"{self.predictor_name}", log_level=logging.DEBUG, queue_handler=True
            )
        else:
            self.logger = LoggingUtility().get_logger(
                f"{logger.name}-{self.predictor_name}", log_level=logging.DEBUG, queue_handler=True
            )

        self.logger.info(f"Initializing {self.predictor_name} predictor")

        self._read_config()
        self.configure()

    @property
    def predictor_name(self):
        return self._predictor_name

    @property
    def aac_dataset(self):
        return self._aac_dataset

    @property
    def database(self):
        return os.path.join(self._personalized_resources_path, self._database)

    @property
    def deltas(self) -> list[float]:
        return [float(delta) for delta in self._deltas.split()]

    @deltas.setter
    def deltas(self, value):
        self._deltas = value

    @property
    def cardinality(self):
        return len(self.deltas)

    @property
    def generic_phrases(self):
        return os.path.join(self._personalized_resources_path, self._generic_phrases)

    @property
    def learn_enabled(self):
        return self._learn

    @property
    def modelname(self):
        return os.path.join(self._static_resources_path, self._modelname)

    @property
    def personalized_cannedphrases(self):
        return os.path.join(self._personalized_resources_path, self._personalized_cannedphrases)

    @property
    def predictor_class(self):
        return self._predictor_class

    @property
    def retrieveaac(self):
        return self._retrieveaac

    @property
    def sbertmodel(self):
        return self._sbertmodel

    @property
    def sentence_transformer_model(self):
        return os.path.join(self._personalized_resources_path, self._sentence_transformer_model)

    @property
    def sent_database(self):
        return os.path.join(self._personalized_resources_path, self._sent_database)

    @property
    def retrieve_database(self):
        return os.path.join(self._static_resources_path, self._retrieve_database)

    @property
    def blacklist_file(self):
        return os.path.join(self._static_resources_path, self._blacklist_file)

    @property
    def embedding_cache_path(self):
        return os.path.join(self._personalized_resources_path, self._embedding_cache_path)

    @property
    def index_path(self):
        return os.path.join(self._personalized_resources_path, self._index_path)

    @property
    def stopwordsFile(self):
        return os.path.join(self._static_resources_path, self._stopwords)

    @property
    def personalized_allowed_toxicwords_file(self):
        return os.path.join(
            self._personalized_resources_path, self._personalized_allowed_toxicwords_file
        )

    @property
    def startsents(self):
        return os.path.join(self._personalized_resources_path, self._startsents)

    @property
    def tokenizer(self):
        return os.path.join(self._static_resources_path, self._tokenizer)

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    @property
    def test_generalsentenceprediction(self):
        return self._test_generalsentenceprediction

    @test_generalsentenceprediction.setter
    def test_generalsentenceprediction(self, value):
        self._test_generalsentenceprediction = value

    @abstractmethod  # pragma: no cover
    def configure(self) -> None:
        raise NotImplementedError(f"Configure not implemented in {self.predictor_name}")

    @abstractmethod  # pragma: no cover
    def predict(
        self, max_partial_prediction_size=None, filter=None
    ) -> tuple[Prediction, Prediction]:
        """
        Predicts the next word and sentence based on the context
        args:
            max_partial_prediction_size: int    # Maximum number of partial predictions to return
            filter: str | None                  # Filter to apply to the predictions

        returns:
            tuple[Prediction, Prediction]       # The predicted next word and sentence
        """
        raise NotImplementedError(f"Configure not implemented in {self.predictor_name}")

    def read_personalized_corpus(self):
        corpus = []

        with open(self.personalized_cannedphrases) as f:
            corpus = f.readlines()
            corpus = [s.strip() for s in corpus]

        return corpus

    def learn(self, change_tokens=None):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def recreate_database(self):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def load_model(*args, **kwargs):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def read_personalized_toxic_words(self, *args, **kwargs):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def _read_config(self):
        try:
            self.logger.debug(f"Reading config for {self.predictor_name}")

            # Pull the predictor_name section from the config

            if self.config.has_section(self.predictor_name):

                properties = {k: v for k, v in vars(self).items() if k.startswith("_")}

                for attr, default in properties.items():
                    option = attr[1:]
                    if option in self.config.options(self.predictor_name):
                        new_value = default
                        try:
                            new_value = self.config.getboolean(
                                self.predictor_name, option, fallback=default
                            )
                        except ValueError:
                            try:
                                new_value = self.config.getint(
                                    self.predictor_name, option, fallback=default
                                )
                            except ValueError:
                                new_value = self.config.get(
                                    self.predictor_name, option, fallback=default
                                )

                        setattr(self, attr, new_value)

        except Exception as e:
            self.logger.error(f"Exception in SentenceCompletionPredictor._read_config = {e}")
            raise e

    def __repr__(self):
        return f"{self.__class__.__name__}({self.predictor_name})"
