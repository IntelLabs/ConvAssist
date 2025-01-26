# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os

from convassist.predictor.utilities.nlp import NLP
from convassist.predictor.utilities.prediction import Prediction, Suggestion
from convassist.utilities.ngram.ngram_map import NgramMap

from .smoothed_ngram_predictor import SmoothedNgramPredictor


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

    # # Override default properties
    # @property
    # def sentences_db(self):
    #     return os.path.join(self._personalized_resources_path, self._sentences_db)

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

    # def recreate_database(self):

    #     # STEP 1: CREATE CANNED_NGRAM DATABASE IF IT DOES NOT EXIST
    #     try:
    #         assert self.ngram_db_conn
    #         self.ngram_db_conn.connect()
    #         for i in range(self.cardinality):
    #             self.ngram_db_conn.create_ngram_table(cardinality=i + 1)

    #     except Exception as e:
    #         self.logger.error(f"exception in creating personalized db : {e}")

    #     phrases_toAdd = self.canned_data.all_phrases_as_list()
    #     phrases_toRemove = []

    #     # Add phrases_toAdd to the ngram database
    #     for phrase in phrases_toAdd:
    #         for curr_card in range(self.cardinality):
    #             ngram_map = NgramMap(curr_card, phrase)

    #             # for every ngram, get db count, update or insert
    #             for ngram, count in ngram_map.items():
    #                 self.ngram_db_conn.insert_ngram(
    #                     curr_card + 1, ngram, count, update_on_conflict=False
    #                 )

    #     for phrase in phrases_toRemove:
    #         for curr_card in range(self.cardinality):
    #             # imp_words = self.extract_svo(phrase)
    #             ngram_map = NgramMap(curr_card, phrase)

    #             # for every ngram, get db count, update or insert
    #             for ngram, count in ngram_map.items():
    #                 self.ngram_db_conn.remove_ngram(ngram)

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

    # TODO: Refactor this class and general_word since this is the same code.
    def predict(self, max_partial_prediction_size: int, filter):
        """
        Predicts the next word based on the context tracker and the n-gram model.
        """
        sentence_predictions = Prediction()  # Not used in this predictor
        word_predictions = Prediction()

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

            return sentence_predictions, word_predictions

        else:
            return super().predict(max_partial_prediction_size, filter)
