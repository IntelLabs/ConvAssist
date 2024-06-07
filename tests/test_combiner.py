"""
 Copyright (C) 2023 Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
import os
import unittest
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
    
import convAssist.word_sentence_predictor
import convAssist.combiner
import convAssist.context_tracker
import convAssist.callback

class StringStreamCallback(convAssist.callback.Callback):
    def __init__(self, stream):
        convAssist.callback.Callback.__init__(self)
        self.stream = stream


class TestMeritocracyCombiner(unittest.TestCase):
    def setUp(self):
        self.combiner = convAssist.combiner.MeritocracyCombiner()
        
        config_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "test_data", "profile_smoothedngram.ini"
            )
        )
        config = configparser.ConfigParser()
        config.read(config_file)
        # config.set("Database", "database", self.dbfilename)

        self.predictor_registry = convAssist.word_sentence_predictor.PredictorRegistry(config)
        
        self.callback = StringStreamCallback("")
        context_tracker = convAssist.context_tracker.ContextTracker(
            config, self.predictor_registry, self.callback
        )


    def _create_prediction(self):
        prediction = convAssist.word_sentence_predictor.Prediction()
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.3, self.predictor_registry))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3, self.predictor_registry))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.1, self.predictor_registry))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2, self.predictor_registry))
        return prediction

    def _create_prediction2(self):
        prediction = convAssist.word_sentence_predictor.Prediction()
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3, self.predictor_registry))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.1, self.predictor_registry))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2, self.predictor_registry))
        return prediction

    def test_filter(self):
        result = self.combiner.filter(self._create_prediction())

        correct = convAssist.word_sentence_predictor.Prediction()
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2,self.predictor_registry))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3,self.predictor_registry))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.4,self.predictor_registry))

        assert result == correct

    def test_combine(self):
        predictions = [self._create_prediction2()]
        prediction2 = self._create_prediction2()
        prediction2.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test4", 0.1,self.predictor_registry))
        predictions.append(prediction2, self.context_tracker)
        result = self.combiner.combine(predictions, convAssist.context_tracker.past_stream())

        correct = convAssist.word_sentence_predictor.Prediction()
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.4,self.predictor_registry))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.6,self.predictor_registry))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test4", 0.1,self.predictor_registry))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.2,self.predictor_registry))

        assert result == correct
