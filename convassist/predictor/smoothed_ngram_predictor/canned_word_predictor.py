# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os

from tqdm import tqdm

from convassist.predictor.utilities.nlp import NLP
from convassist.predictor.utilities import Predictions, Suggestion, PredictorResponses
from convassist.utilities.ngram.ngram_map import NgramMap
from convassist.utilities.ngram.ngramutil import NGramUtil

from convassist.predictor.smoothed_ngram_predictor.smoothed_ngram_predictor import SmoothedNgramPredictor


class CannedWordPredictor(SmoothedNgramPredictor):
    """
    CannedWordPredictor is a specialized predictor that extends the SmoothedNgramPredictor.
    It is designed to handle canned responses using natural language processing (NLP) techniques.

    Methods:
        configure():
            Configures the predictor by loading the NLP model and initializing constants and stopwords.

        extract_svo(sent: str) -> str:
            Extracts significant subject-verb-object (SVO) tokens from a given sentence.

        recreate_database():
            Recreates the sentence and n-gram databases by adding new phrases and removing outdated ones.

    """

    def configure(self):
        # load the natural language processing model
        self.nlp = NLP().get_nlp()

        # object and subject constants
        self.OBJECT_DEPS = {
            "dobj",
            "pobj",
            "dative",
            "attr",
            "oprd",
            "npadvmod",
            "amod",
            "acomp",
            "advmod",
        }
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwordsList = []

        with open(self.stopwordsFile) as f:
            self.stopwordsList = f.read().splitlines()

            # strip each word in stopwordsList
            self.stopwordsList = [word.strip() for word in self.stopwordsList]

        super().configure()


    def extract_svo(self, sent):
        doc = self.nlp(sent)
        sub = []
        at = []
        ve = []
        imp_tokens = []
        for token in doc:
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
                if (
                    token.text.lower() not in self.stopwordsList
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if (
                    token.text.lower() not in self.stopwordsList
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if (
                    token.text.lower() not in self.stopwordsList
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
        return " ".join(imp_tokens).strip().lower()

    def recreate_database(self):
        """
        Recreates the sentence and n-gram databases by adding new phrases and removing outdated ones.
        """
        try:
            with open(self.personalized_cannedphrases) as f:
                phrases = f.read().splitlines()

            for cardinality in range(1, self.cardinality + 1):

                with NGramUtil(self.database, self.cardinality) as ngramutil:

                    query = ngramutil.generate_ngram_insert_query(cardinality, False)

                    data = []
                    for phrase in tqdm(phrases, desc=f"Processing {cardinality}-grams", unit=" phrases", leave=False):
                        ngram_map = NgramMap(cardinality, phrase)
                        for ngram, count in ngram_map.items():
                            data.append((*ngram, count))

                    batch_size = 3000
                    for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting {cardinality}-grams", unit=" batches", leave=True):
                        batch = data[i:i + batch_size]
                        ngramutil.connection.execute_many(query, batch)

        except Exception as e:
            self.logger.error(f"exception in creating personalized db : {e}")

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    # TODO: Refactor this class and general_word since this is the same code.
    def predict(self, max_partial_prediction_size: int, filter) -> PredictorResponses:
        """
        Predicts the next word based on the context tracker and the n-gram model.
        """
        responses = PredictorResponses()
        word_predictions = responses.word_predictions

        actual_tokens, _ = self.context_tracker.get_tokens(self.cardinality)

        if actual_tokens == 0:
            self.logger.warning(
                f"No tokens in the context tracker.  Getting {max_partial_prediction_size} most frequent start words..."
            )

            with open(self.startwords) as f:
                self.precomputed_StartWords = json.load(f)

            for w, prob in list(self.precomputed_StartWords.items())[:max_partial_prediction_size]:
                word_predictions.add_suggestion(Suggestion(w, prob, self.predictor_name))

            if len(word_predictions) == 0:
                self.logger.error("Error getting most frequent start words.")

            return responses

        else:
            return super().predict(max_partial_prediction_size, filter)
