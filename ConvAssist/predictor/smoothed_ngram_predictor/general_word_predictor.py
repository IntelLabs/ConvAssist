# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import json
import os

from ..utilities.prediction import Prediction, Suggestion
from .smoothed_ngram_predictor import SmoothedNgramPredictor


class GeneralWordPredictor(SmoothedNgramPredictor):
    """
    Calculates prediction from n-gram model in sqlite database.

    """

    def configure(self):
        super().configure()

        # Store the set of most frequent starting words based on an AAC dataset
        # These will be displayed during empty context
        if not os.path.isfile(self.startwords):
            aac_lines = open(self.aac_dataset).readlines()
            startwords = []
            for line in aac_lines:
                w = line.lower().split()[0]
                startwords.append(w)
            counts = collections.Counter(startwords)
            total = sum(counts.values())
            self.precomputed_StartWords = {k: v / total for k, v in counts.items()}
            with open(self.startwords, "w") as fp:
                json.dump(self.precomputed_StartWords, fp)

    # Override default properties
    @property
    def aac_dataset(self):
        return os.path.join(self._static_resources_path, self._aac_dataset)

    @property
    def database(self):
        return os.path.join(self._static_resources_path, self._database)

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    def predict(self, max_partial_prediction_size: int, filter):
        """
        Predicts the next word based on the context tracker and the n-gram model.
        """
        sentence_predictions = Prediction()  # Not used in this predictor
        word_predictions = Prediction()

        actual_tokens, _ = self.context_tracker.get_tokens(self.cardinality)

        if actual_tokens == 0:
            self.logger.warning(
                "No tokens in the context tracker.  Getting most frequent start words..."
            )

            with open(self.startwords) as f:
                self.precomputed_StartWords = json.load(f)

            for w, prob in list(self.precomputed_StartWords.items())[:max_partial_prediction_size]:
                word_predictions.add_suggestion(Suggestion(w, prob, self.predictor_name))

            if len(word_predictions) == 0:
                self.logger.error("Error getting most frequent start words.")

            return sentence_predictions, word_predictions

        else:
            return super().predict(max_partial_prediction_size, filter)
