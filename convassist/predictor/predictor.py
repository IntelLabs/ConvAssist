# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
from abc import ABC, abstractmethod
from configparser import ConfigParser

from convassist.context_tracker import ContextTracker
from convassist.predictor.utilities import PredictorResponses
from convassist.utilities.logging_utility import LoggingUtility


class Predictor(ABC):
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

        # configure a logger
        if logger is None:
            self.logger = LoggingUtility().get_logger(
                f"{self.predictor_name}", log_level=logging.DEBUG, queue_handler=True
            )
        else:
            self.logger = LoggingUtility().get_logger(
                f"{logger.name}-{self.predictor_name}", log_level=logger.level, queue_handler=True
            )

        self.logger.info(f"Initializing {self.predictor_name} predictor")

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

        self._read_config()

        self.logger.info(f"Finished initializing {self.predictor_name} predictor")

        # TODO: FIXME - Change functionality to force a call to configure.
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
        return self._modelname

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
    
    @sbertmodel.setter
    def sbertmodel(self, value):
        self._sbertmodel = value

    @property
    def sentence_transformer_model(self):
        return self._sentence_transformer_model

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
        return self._tokenizer

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
    ) -> PredictorResponses:
        """
        Predicts the next word and sentence based on the context
        args:
            max_partial_prediction_size: int    # Maximum number of partial predictions to return
            filter: str | None                  # Filter to apply to the predictions

        returns:
            tuple[Prediction, Prediction]       # The predicted next word and sentence
        """
        raise NotImplementedError(f"Configure not implemented in {self.predictor_name}")

    def learn(self, change_tokens=None):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def recreate_database(self):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def load_model(self):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def read_personalized_toxic_words(self, *args, **kwargs):  # pragma: no cover
        # Not all predictors need this, but define it here for those that do
        pass

    def _find_option_in_section(self, option: str, section: str) -> str:
        if self.config.has_option(section, option):
            return section

        # TODO: Document this behavior
        elif self.config.has_option("Common", option):
            return "Common"

        else:
            return ""

    def _read_config(self):
        try:
            self.logger.debug(f"Reading config for {self.predictor_name}")

            # Pull the predictor_name section from the config

            if self.config.has_section(self.predictor_name):

                properties = {k: v for k, v in vars(self).items() if k.startswith("_")}

                for attr, default in properties.items():
                    option = attr[1:]
                    section = self._find_option_in_section(option, self.predictor_name)

                    new_value = default
                    if section:
                        try:
                            new_value = self.config.getboolean(section, option, fallback=default)
                        except ValueError:
                            try:
                                new_value = self.config.getint(section, option, fallback=default)
                            except ValueError:
                                new_value = self.config.get(section, option, fallback=default)

                    setattr(self, attr, new_value)

        except Exception as e:
            self.logger.error(f"Exception in SentenceCompletionPredictor._read_config = {e}")
            raise e

    def __repr__(self):
        return f"{self.__class__.__name__}({self.predictor_name})"
