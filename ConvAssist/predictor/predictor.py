# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging
from configparser import ConfigParser
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


class Predictor(metaclass=OptionalSingleton):
    """
    Base class for predictors.

    """

    def __init__(
            self, 
            config: ConfigParser, 
            context_tracker: ContextTracker, 
            predictor_name: str, 
            short_desc: str | None = None, 
            long_des: str | None = None,
            logger: logging.Logger | None = None
    ):
        self.config = config
        self.context_tracker = context_tracker
        self._predictor_name = predictor_name
        self._short_description = short_desc
        self._long_description = long_des
        
        #configure a logger
        if logger:
            self.logger = logger
        else:
            self.logger = LoggingUtility().get_logger(self.predictor_name, log_level=logging.DEBUG)
            
    @property
    def predictor_name(self):
        return self._predictor_name
    
    @property
    def short_description(self):
        return self._short_description
    
    @property
    def long_description(self):
        return self._long_description

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
        pass
    
    @abstractmethod
    def learn(self, change_tokens = None):
        '''
        Learns from the context
        args:
            change_tokens: list[str] | None                 # List of tokens to learn from
        '''
        pass

    @abstractmethod
    def _read_config(self):
        '''
        Reads the configuration file
        '''
        pass
    

    def load_model(*args, **kwargs):
        # Not all predictors need this, but define it here for those that do
        pass

    def read_personalized_toxic_words(self, *args, **kwargs):
        # Not all predictors need this, but define it here for those that do
        pass
        
        
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

