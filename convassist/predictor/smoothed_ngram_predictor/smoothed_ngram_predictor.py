# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import string
from abc import ABC
from typing import List

from convassist.predictor import Predictor
from convassist.predictor.utilities.prediction import Prediction, Suggestion
from convassist.utilities.ngram.ngramutil import NGramUtil


class SmoothedNgramPredictor(Predictor, ABC):
    """
    SmoothedNgramPredictor is a class that extends the Predictor class to provide
    functionality for predicting the next word(s) in a sequence using smoothed n-grams.
    """

    def configure(self) -> None:
        # # make sure personalized databases are updated
        # with NGramUtil(self.database, cardinality=3) as ngramutil:
        #     with open(self.personalized_cannedphrases, "r") as f:
        #         for line in f:
        #             ngramutil.learn(line.strip('.\n'))
        pass

    def extract_svo(self, sent):
        return sent

    def predict(self, max_partial_prediction_size: int, filter):

        sentence_prediction = Prediction()
        word_prediction = Prediction()

        self.logger.debug("Starting Ngram prediction")

        # get self.cardinality tokens from the context tracker
        actual_tokens, tokens = self.context_tracker.get_tokens(self.cardinality)
        prefix_completion_candidates: List[str] = []

        try:
            partial = None
            prefix_ngram = None
            # for ngram_len in reversed(range(1, actual_tokens + 1)):
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
                        # candidate = p[-2]
                        # if (
                        #     candidate not in tokens
                        #     and candidate not in prefix_completion_candidates
                        # ):
                        #     prefix_completion_candidates.append(candidate)
                        prefix_completion_candidates.append(p[-2])

                        # smoothing
                        unigram_counts_sum = ngramutil.unigram_counts_sum()

                        candidate_tokens = [""] * ngram_len
                        for i in range(actual_tokens):
                            candidate_tokens[i] = tokens[i]

                        for j, candidate in enumerate(prefix_completion_candidates):
                            candidate_tokens[ngram_len - 1] = candidate
                            probability = 0.0
                            for k in range(ngram_len):
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

        if len(word_prediction) == 0:
            self.logger.error(f"No predictions from {self.predictor_name}")

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
                    self.logger.debug(f"learning ...{phrase}")

                    phrase = phrase.lower().translate(str.maketrans("", "", string.punctuation))
                    phrase = self.extract_svo(phrase)

                    ngramutil.learn(phrase)

                except Exception as e:
                    self.logger.error(f"{self.predictor_name} learn function: {e}")

        pass
