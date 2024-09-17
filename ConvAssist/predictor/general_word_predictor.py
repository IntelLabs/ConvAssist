
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from configparser import ConfigParser
import logging
import os
import collections
import json
import string
from pathlib import Path
from typing import Any, List, Optional

from ConvAssist.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.predictor.utilities.suggestion import Suggestion

class GeneralWordPredictor(SmoothedNgramPredictor):
    """
    Calculates prediction from n-gram model in sqlite database.

    """

    def __init__(
            self,
            config: ConfigParser,
            context_tracker: ContextTracker,
            predictor_name: str,
            logger: logging.Logger | None = None
    ):
        super().__init__(
            config, context_tracker,
            predictor_name, logger=logger
        )

        # Store the set of most frequent starting words based on an AAC dataset
        # These will be displayed during empty context
        if(not os.path.isfile(self.startwords)):
            aac_lines = open(self.aac_dataset,"r").readlines()
            startwords = []
            for line in aac_lines:
                w = line.lower().split()[0]
                startwords.append(w)
            counts = collections.Counter(startwords)
            total = sum(counts.values())
            self.precomputed_sentenceStart = {k:v/total for k,v in counts.items()}
            with open(self.startwords, 'w') as fp:
                json.dump(self.precomputed_sentenceStart, fp)

    #Override default properties
    @property
    def aac_dataset(self):
        return os.path.join(self._static_resources_path, self._aac_dataset)

    @property
    def database(self):
        return os.path.join(self._static_resources_path, self._database)

    @database.setter
    def database(self, value):
        self._database = value
        self.init_database_connector_if_ready()

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    def predict(self, max_partial_prediction_size: int, filter):
        '''
        Predicts the next word based on the context tracker and the n-gram model.
        '''
        sentence_prediction = Prediction() # Not used in this predictor
        word_prediction = Prediction()

        actual_tokens, tokens = self.context_tracker.get_tokens(self.cardinality)

        if actual_tokens == 0 or not tokens or tokens[0] == " ":
            self.logger.warning(f"No tokens in the context tracker.  Getting most frequent start words...")

            with open(self.startwords) as f:
                self.precomputed_sentenceStart = json.load(f)

            for w, prob in self.precomputed_sentenceStart.items():
                word_prediction.add_suggestion(
                        Suggestion(w, prob, self.predictor_name)
                    )

            if len(word_prediction) == 0:
                self.logger.error(f"Error getting most frequent start words.")

            return sentence_prediction, word_prediction

        else:
            return super().predict(max_partial_prediction_size, filter)
