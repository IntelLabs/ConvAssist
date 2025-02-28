# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import string
from abc import ABC
from typing import List

from convassist.predictor.predictor import Predictor
from convassist.predictor.utilities.prediction import Prediction, Suggestion
from convassist.utilities.ngram.ngramutil import NGramUtil


class SmoothedNgramPredictor(Predictor):
    """
    SmoothedNgramPredictor is a class that extends the Predictor class to provide
    functionality for predicting the next word(s) in a sequence using smoothed n-grams.
    """

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    def configure(self) -> None:
        with NGramUtil(self.database, self.cardinality) as ngramutil:
            try:
                ngramutil.create_update_ngram_tables()

            except Exception as e:
                self.logger.error(f"Error creating ngram tables: {e}")

    def extract_svo(self, sent):
        return sent
    
    def get_frequent_start_words(self, max_count = 10) -> Prediction:
        word_predictions = Prediction()

        try:
            with open(self.startwords) as f:
                self.precomputed_StartWords = json.load(f)

            for w, prob in list(self.precomputed_StartWords.items())[:max_count]:
                word_predictions.add_suggestion(Suggestion(w, prob, self.predictor_name))

        except FileNotFoundError:
            self.logger.info(f"No frequent start words present.")

        return word_predictions

    def predict(self, max_partial_prediction_size: int, filter) -> tuple[Prediction, Prediction]:

        sentence_prediction = Prediction()
        word_prediction = Prediction()

        self.logger.debug("Starting Ngram prediction")

        # get self.cardinality tokens from the context tracker
        actual_tokens, tokens = self.context_tracker.get_tokens(self.cardinality)
        prefix_completion_candidates: List[str] = []

        if actual_tokens == 0:
            self.logger.info(
                f"No tokens in the context tracker.  Getting {max_partial_prediction_size} most frequent start words if supported."
            )
            word_prediction = self.get_frequent_start_words(max_count=max_partial_prediction_size)

        else:
            self.logger.debug(f"Actual tokens: {actual_tokens}, tokens: {tokens}")
            try:
                partial = None
                prefix_ngram = None

                for ngram_len in range(actual_tokens, 0, -1):
                    if len(prefix_completion_candidates) >= max_partial_prediction_size:
                        break

                    with NGramUtil(self.database, ngram_len) as ngramutil:

                        if ngram_len:
                            prefix_ngram = tokens[-(ngram_len):]
                        else:
                            # just get the last token
                            prefix_ngram = tokens[-1:]

                        if prefix_ngram:
                            try:
                                partial = ngramutil.fetch_like(
                                    prefix_ngram,
                                    max_partial_prediction_size - len(prefix_completion_candidates),
                                )
                            except Exception as e:
                                self.logger.error(f"Error fetching ngrams for {prefix_ngram}: {e}")
                                continue
                        for p in partial:
                            prefix_completion_candidates.append(p[-2])

                        # smoothing
                        unigram_counts_sum = ngramutil.unigram_counts_sum()

                        candidate_tokens = prefix_ngram.copy()

                        for j, candidate in enumerate(prefix_completion_candidates):
                            candidate_tokens[-1] = candidate
                            probability = 0.0
                            for k in range(len(candidate_tokens)):
                                numerator = ngramutil.count(candidate_tokens, 0, k + 1)

                                denominator = unigram_counts_sum
                                if numerator > 0:
                                    denominator = ngramutil.count(candidate_tokens, -1, k)
                                frequency = 0
                                if denominator > 0:
                                    frequency = float(numerator) / denominator
                                probability += float(self.deltas[k]) * frequency
                            if probability > 0:
                                if all(
                                    char in string.punctuation
                                    for char in candidate_tokens[ngram_len - 1]
                                ):
                                    self.logger.debug(
                                        candidate_tokens[ngram_len - 1] + " contains punctuations "
                                    )
                                else:
                                    word_prediction.add_suggestion(
                                        Suggestion(
                                            candidate_tokens[ngram_len - 1],
                                            probability,
                                            self.predictor_name,
                                        )
                                    )
            except Exception as e:
                self.logger.error(f"Exception in {self.predictor_name} predict function: {e}")

            self.logger.info(
                f"End prediction. got {len(word_prediction)} word suggestions and {len(sentence_prediction)} sentence suggestions"
            )

        return sentence_prediction, word_prediction

    def learn(self, phrase):
        # build up ngram map for all cardinalities
        # i.e. learn all ngrams and counts in memory
        if self.learn_enabled:
            with NGramUtil(self.database, self.cardinality) as ngramutil:
                try:

                    phrase = phrase.lower().translate(str.maketrans("", "", string.punctuation))
                    phrase = self.extract_svo(phrase)
                    self.logger.debug(f"learning ... {phrase}")

                    ngramutil.learn(phrase)

                except Exception as e:
                    self.logger.error(f"{self.predictor_name} learn function: {e}")

        pass
