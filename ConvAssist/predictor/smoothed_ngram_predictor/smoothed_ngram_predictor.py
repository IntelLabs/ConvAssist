# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import string
from typing import List

from ...utilities.databaseutils.sqllite_ngram_dbconnector import (
    SQLiteNgramDatabaseConnector,
)
from ..predictor import Predictor
from ..utilities.prediction import Prediction, Suggestion
from .ngram_map import NgramMap


class SmoothedNgramPredictor(Predictor):
    """
    SmoothedNgramPredictor is a class that extends the Predictor class to provide
    functionality for predicting the next word(s) in a sequence using smoothed n-grams.

    Methods:
        configure() -> None:
            Configures the predictor by initializing the n-gram database connector and recreating the database.

        recreate_database():
            Placeholder method for recreating the database.

        extract_svo(sent):
            Default implementation that returns the input sentence.

        generate_ngrams(token, n):
            Generates n-grams from the given tokens.

        getNgramMap(ngs, ngram_map):
            Populates the n-gram map with the given n-grams.

        predict(max_partial_prediction_size: int, filter):
            Predicts the next word(s) based on the context and n-gram database.

        _count(tokens, offset, ngram_size):
            Counts the occurrences of the specified n-gram in the database.

        learn(change_tokens):
            Learns new n-grams from the given tokens and updates the n-gram database.
    """

    def configure(self) -> None:
        self.ngram_db_conn = SQLiteNgramDatabaseConnector(self.database, self.cardinality)

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

        if actual_tokens == 0 or not tokens:
            self.logger.warning("No tokens in the context tracker.")
            return sentence_prediction, word_prediction

        try:
            assert self.ngram_db_conn is not None
            self.ngram_db_conn.connect()

            prefix_completion_candidates: List[str] = []

            partial = None
            prefix_ngram = None
            for ngram_len in reversed(range(actual_tokens)):
                if ngram_len:
                    prefix_ngram = tokens[-(ngram_len - 1) :]
                else:
                    # just get the last token
                    prefix_ngram = tokens[-1:]

                if prefix_ngram:
                    partial = self.ngram_db_conn.ngram_fetch_like(
                        prefix_ngram,
                        max_partial_prediction_size - len(prefix_completion_candidates),
                    )

                if partial:
                    for p in partial:
                        if len(prefix_completion_candidates) >= max_partial_prediction_size:
                            break
                        candidate = p[-2]
                        if (
                            candidate not in tokens
                            and candidate not in prefix_completion_candidates
                        ):
                            prefix_completion_candidates.append(candidate)

                if len(prefix_completion_candidates) >= max_partial_prediction_size:
                    break

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
