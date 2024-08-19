# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from src.utilities.logging import ConvAssistLogger
from src.utilities.singleton import PredictorSingleton
from src.predictor.utilities.prediction import Prediction
from src.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector

class Predictor(metaclass=PredictorSingleton):
    """
    Base class for predictors.

    """

    def __init__(
            self, 
            config, 
            context_tracker, 
            predictor_name, 
            short_desc=None, 
            long_desc=None,
                        logger=None
    ):
        self.config = config
        self.context_tracker = context_tracker
        self.predictor_name = predictor_name
        self.short_description = short_desc
        self.long_description = long_desc
        
            
        #configure a logger
        if logger:
            self.log = logger
        else:
            self.log = ConvAssistLogger(name=self.predictor_name, 
                                        level="DEBUG")
    
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
            self.log.critical(f"Unable to create table {tablename} in {dbname}:  {e}")
            raise e

