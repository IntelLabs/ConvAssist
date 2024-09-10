# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging
from configparser import ConfigParser
import os
from ConvAssist.utilities.logging_utility import LoggingUtility
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector

class OptionalSingleton(type):
    """
    A metaclass that allows the class to be a singleton if the singleton attribute is set to True.
    """
    _instances = {}
    _singleton_enabled = False

    def __call__(cls, *args, **kwargs):
        if cls._singleton_enabled:
            if cls not in cls._instances:
                cls._instances[cls] = super(OptionalSingleton, cls).__call__(*args, **kwargs)
            return cls._instances[cls]
        else:
            return super(OptionalSingleton, cls).__call__(*args, **kwargs)
        
    @classmethod
    def enable_singleton(cls):
        cls._singleton_enabled = True


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
    - createTable(self, dbname, tablename, columns) -> None
        Creates a table in the database.
    """

    def __init__(
            self, 
            config: ConfigParser, 
            context_tracker: ContextTracker, 
            predictor_name: str, 
            logger: logging.Logger | None = None
    ):

        self.config = config
        self.context_tracker = context_tracker
        self._predictor_name = predictor_name
        
        #configure a logger
        if logger:
            self.logger = logger
        else:
            self.logger = LoggingUtility().get_logger(self.predictor_name, log_level=logging.DEBUG)

        self.logger.info(f"Initializing {self.predictor_name} predictor")
            
    @property
    def predictor_name(self):
        return self._predictor_name
    
    @abstractmethod    
    def predict(self, max_partial_prediction_size = None, filter = None)  -> tuple[Prediction, Prediction]:
        '''
        Predicts the next word and sentence based on the context
        args:
            max_partial_prediction_size: int    # Maximum number of partial predictions to return
            filter: str | None                  # Filter to apply to the predictions

        returns:
            tuple[Prediction, Prediction]       # The predicted next word and sentence
        '''
        self.logger.info(f"{self.predictor_name} - Predicting next word and sentence")
    
    @abstractmethod
    def _read_config(self):
        '''
        Reads the configuration file
        '''
        pass

    def read_personalized_corpus(self):
        corpus = []
        path = self.config.get(self.predictor_name, "personalized_resources_path")
        personalized_cannedphrases = os.path.join(path, self.predictor_name, "personalized_cannedphrases")

        if os.path.exists(personalized_cannedphrases):
            corpus = open(personalized_cannedphrases, "r").readlines()
            corpus = [s.strip() for s in corpus]

        return corpus

    
    def learn(self, change_tokens = None):
        # Not all predictors need this, but define it here for those that do
        pass

    def recreate_database(self):
        # Not all predictors need this, but define it here for those that do
        pass

    def load_model(*args, **kwargs):
        # Not all predictors need this, but define it here for those that do
        pass

    def is_model_loaded(self):
        # Not all predictors need this, but define it here for those that do
        pass

    def read_personalized_toxic_words(self, *args, **kwargs):
        # Not all predictors need this, but define it here for those that do
        pass
        
    #TODO: Move this to a better place. This is a utility function and should not be in the Predictor class
    def createTable(self, dbname, tablename, columns):
        try:
            conn = SQLiteDatabaseConnector(dbname)
            conn.connect()
    
            conn.execute_query(f'''
                CREATE TABLE IF NOT EXISTS {tablename}
                ({', '.join(columns)})
            ''')
            conn.close()
        except Exception as e:
            raise Exception(f"Unable to create table {tablename} in {dbname}.", e)

