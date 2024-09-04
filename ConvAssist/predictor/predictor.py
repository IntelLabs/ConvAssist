# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from configparser import ConfigParser
from ConvAssist.utilities.logging_utility import LoggingUtility
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector

class Predictor():
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
        self.predictor_name = predictor_name
        self.short_description = short_desc
        self.long_description = long_des
        
        #configure a logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.predictor_name)
            self.logger.configure_logging(level="DEBUG") #type: ignore
    def get_name(self):
        return self.predictor_name
    
    def get_description(self):
        return self.long_description
    
    def get_long_description(self):
        return self.long_description
    
    def predict(self, max_partial_prediction_size = None, filter = None)  -> tuple[Prediction, Prediction]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def learn(self, change_tokens = None):
        raise NotImplementedError("Subclasses must implement this method")
    
    def _read_config(self):
        raise NotImplementedError("Subclasses must implement this method")
    
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

