# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import string
from typing import List

from ConvAssist.predictor.predictor import Predictor
from ConvAssist.predictor.smoothed_ngram_predictor.ngram_map import NgramMap
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.predictor.utilities.suggestion import Suggestion
from ConvAssist.utilities.databaseutils.sqllite_ngram_dbconnector import (
    SQLiteNgramDatabaseConnector,
)


class SmoothedNgramPredictor(Predictor):
    """
    Calculates prediction from n-gram model in sqlite database.

    """

    def configure(self) -> None:
        self.ngram_db_conn = SQLiteNgramDatabaseConnector(
            self.database, self.cardinality, self.logger
        )

        self.recreate_database()

    def recreate_database(self):
        pass

    # Default implementation
    def extract_svo(self, sent):
        return sent

    # # default implementation
    # def recreate_database(self):
    #     pass

    def generate_ngrams(self, token, n):
        n = n + 1
        # Use the zip function to help us generate n-grams
        # Concatenate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        returnobj = [" ".join(ngram) for ngram in ngrams]
        return returnobj

    def getNgramMap(self, ngs, ngram_map):
        for item in ngs:
            tokens = item.split(" ")
            ngram_list = []
            for token in tokens:
                idx = ngram_map.add_token(token)
                ngram_list.append(idx)
            ngram_map.add(ngram_list)
        return ngram_map

    def predict(self, max_partial_prediction_size: int, filter):

        sentence_prediction = Prediction()
        word_prediction = Prediction()

        # get self.cardinality tokens from the context tracker
        actual_tokens, tokens = self.context_tracker.get_tokens(self.cardinality)

        try:
            assert self.ngram_db_conn is not None
            self.ngram_db_conn.connect()

            prefix_completion_candidates: List[str] = []

            for k in reversed(range(actual_tokens)):
                if len(prefix_completion_candidates) >= max_partial_prediction_size:
                    break
                prefix_ngram = tokens[(len(tokens) - k - 1) :]

                if prefix_ngram:
                    partial = self.ngram_db_conn.ngram_fetch_like(
                        prefix_ngram,
                        max_partial_prediction_size - len(prefix_completion_candidates),
                    )

                if partial:
                    for p in partial:
                        if len(prefix_completion_candidates) >= max_partial_prediction_size:
                            break
                        # TODO explain why -2
                        candidate = p[-2]
                        if candidate not in prefix_completion_candidates:
                            prefix_completion_candidates.append(candidate)

            # smoothing
            unigram_counts_sum = self.ngram_db_conn.unigram_counts_sum()

            candidate_tokens = [""] * self.cardinality
            for i in range(actual_tokens):
                candidate_tokens[i] = tokens[i]

            for j, candidate in enumerate(prefix_completion_candidates):
                candidate_tokens[self.cardinality - 1] = candidate
                probability = 0.0
                for k in range(self.cardinality):
                    numerator = self._count(candidate_tokens, 0, k + 1)

                    denominator = unigram_counts_sum
                    if numerator > 0:
                        denominator = self._count(candidate_tokens, -1, k)
                    frequency = 0
                    if denominator > 0:
                        frequency = float(numerator) / denominator
                    probability += float(self.deltas[k]) * frequency
                if probability > 0:
                    if all(
                        char in string.punctuation
                        for char in candidate_tokens[self.cardinality - 1]
                    ):
                        self.logger.debug(
                            candidate_tokens[self.cardinality - 1] + " contains punctuations "
                        )
                    else:
                        word_prediction.add_suggestion(
                            Suggestion(
                                candidate_tokens[self.cardinality - 1],
                                probability,
                                self.predictor_name,
                            )
                        )
        except Exception as e:
            self.logger.error(f"Exception in {self.predictor_name} predict function: {e}")

        finally:
            if self.ngram_db_conn is not None:
                self.ngram_db_conn.close

        if len(word_prediction) == 0:
            self.logger.error(f"No predictions from {self.predictor_name}")

        self.logger.info(
            f"End prediction. got {len(word_prediction)} word suggestions and {len(sentence_prediction)} sentence suggestions"
        )

        return sentence_prediction, word_prediction

    def _count(self, tokens, offset, ngram_size):
        result = 0
        if ngram_size > 0:
            ngram = tokens[len(tokens) - ngram_size + offset : len(tokens) + offset]
            result = self.ngram_db_conn.ngram_count(ngram)
        else:
            result = self.ngram_db_conn.unigram_counts_sum()
        return result

    def learn(self, change_tokens):
        # build up ngram map for all cardinalities
        # i.e. learn all ngrams and counts in memory
        if self.learn_enabled:
            try:
                assert self.ngram_db_conn is not None
                self.ngram_db_conn.connect()

                self.logger.debug("learning ..." + str(change_tokens))
                change_tokens = change_tokens.lower().translate(
                    str.maketrans("", "", string.punctuation)
                )
                self.logger.debug("after removing punctuations, change_tokens = " + change_tokens)

                change_tokens = self.extract_svo(change_tokens)

                change_tokens = change_tokens.split()
                for curr_card in range(self.cardinality):
                    ngram_map = NgramMap()
                    ngs = self.generate_ngrams(change_tokens, curr_card)
                    # ngram_map = self.getNgramMap(ngs, ngram_map)
                    for item in ngs:
                        tokens = item.split(" ")
                        ngram_list = []
                        for token in tokens:
                            idx = ngram_map.add_token(token)
                            ngram_list.append(idx)
                        ngram_map.add(ngram_list)

                    # write this ngram_map to LM ...
                    # for every ngram, get db count, update or insert
                    for ngram, count in ngram_map.items():
                        old_count = self.ngram_db_conn.ngram_count(ngram)
                        if old_count > 0:
                            self.ngram_db_conn.update_ngram(ngram, old_count + count)
                            self.ngram_db_conn.commit()
                        else:
                            self.ngram_db_conn.insert_ngram(ngram, count)
                            self.ngram_db_conn.commit()
            except Exception as e:
                self.logger.error(f"{self.predictor_name} learn function: {e}")

            finally:
                if self.ngram_db_conn is not None:
                    self.ngram_db_conn.close
        pass
