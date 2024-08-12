"""
 Copyright (C) <year(s)> Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
import os
import unittest

import configparser
import convAssist.word_sentence_predictor
import convAssist.tokenizer
import convAssist.dbconnector
import convAssist.context_tracker 
import convAssist.callback


class TestSuggestion(unittest.TestCase):
    def setUp(self):
        self.suggestion = convAssist.word_sentence_predictor.Suggestion("Test", 0.3, "test_predictor")

    def test_probability(self):
        self.suggestion.probability = 0.1
        assert self.suggestion.probability == 0.1


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.prediction = convAssist.word_sentence_predictor.Prediction()

    def test_add_suggestion(self):
        self.prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.3, "test_predictor"))
        assert self.prediction[0].word == "Test"
        assert self.prediction[0].probability == 0.3

        self.prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.2, "test_predictor"))
        assert self.prediction[0].word == "Test"
        assert self.prediction[0].probability == 0.3
        assert self.prediction[1].word == "Test2"
        assert self.prediction[1].probability == 0.2

        self.prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.6, "test_predictor"))
        assert self.prediction[0].word == "Test3"
        assert self.prediction[0].probability == 0.6
        assert self.prediction[1].word == "Test"
        assert self.prediction[1].probability == 0.3
        assert self.prediction[2].word == "Test2"
        assert self.prediction[2].probability == 0.2

        self.prediction[:] = []

    def test_suggestion_for_token(self):
        self.prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Token", 0.8, "test_predictor"))
        assert self.prediction.suggestion_for_token("Token").probability == 0.8
        self.prediction[:] = []


class StringStreamCallback(convAssist.callback.Callback):
    def __init__(self, stream):
        convAssist.callback.Callback.__init__(self)
        self.stream = stream


class TestSmoothedNgramPredictor(unittest.TestCase):
    def setUp(self):
        self.dbfilename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_data", "test.db")
        )
        self.infile = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_data", "der_linksdenker.txt")
        )

        for ngram_size in range(3):
            ngram_map = convAssist.tokenizer.forward_tokenize_file(
                self.infile, ngram_size + 1, False
            )
            convAssist.dbconnector.insert_ngram_map_sqlite(
                ngram_map, ngram_size + 1, self.dbfilename, False
            )

        config_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "test_data", "profile_smoothedngram.ini"
            )
        )
        config = configparser.ConfigParser()
        config.read(config_file)
        config.set("Database", "database", self.dbfilename)

        self.predictor_registry = convAssist.word_sentence_predictor.PredictorRegistry(config)

        self.callback = StringStreamCallback("")
        context_tracker = convAssist.context_tracker.ContextTracker(
            config, self.predictor_registry, self.callback
        )

    def test_predict(self):
        predictor = self.predictor_registry[0]
        predictions = predictor.predict(6, None)
        assert len(predictions) == 6
        words = []
        for p in predictions:
            words.append(p.word)
        assert "er" in words
        assert "der" in words
        assert "die" in words
        assert "und" in words
        assert "nicht" in words

        self.callback.stream = "d"
        predictions = predictor.predict(6, None)
        assert len(predictions) == 6
        words = []
        for p in predictions:
            words.append(p.word)
        assert "der" in words
        assert "die" in words
        assert "das" in words
        assert "da" in words
        assert "Der" in words

        self.callback.stream = "de"
        predictions = predictor.predict(6, None)
        assert len(predictions) == 6
        words = []
        for p in predictions:
            words.append(p.word)
        assert "der" in words
        assert "Der" in words
        assert "dem" in words
        assert "den" in words
        assert "des" in words

    def tearDown(self):
        if self.predictor_registry[0].db:
            self.predictor_registry[0].db.close_database()
        del self.predictor_registry[0]
        if os.path.isfile(self.dbfilename):
            os.remove(self.dbfilename)
