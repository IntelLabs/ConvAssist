# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from configparser import ConfigParser
import os
from pathlib import Path

from ConvAssist.predictor.utilities.prediction import UnknownCombinerException
from ConvAssist.predictor.utilities.predictor_names import PredictorNames
from ConvAssist.combiner.meritocrity_combiner import MeritocracyCombiner

class PredictorActivator(object):
    """
    PredictorActivator starts the execution of the active predictors,
    monitors their execution and collects the predictions returned, or
    terminates a predictor's execution if it execedes its maximum
    prediction time.

    The predictions returned by the individual predictors are combined
    into a single prediction by the active Combiner.

    """

    def __init__(self, config, registry, context_tracker=None):
        self.config:ConfigParser = config
        self.registry = registry
        self.context_tracker = context_tracker
        self.predictions = []
        self.word_predictions = []
        self.sent_predictions = []
        self.spell_word_predictions = []
        self.combiner: MeritocracyCombiner
        self.max_partial_prediction_size = self.config.getint("Selector", "suggestions", fallback=10)
        self.predict_time = None
        self._combination_policy = None

    @property
    def combination_policy(self):
        """The combination_policy property."""
        return self._combination_policy

    @combination_policy.setter
    def combination_policy(self, value):
        self._combination_policy = value
        if value.lower() == "meritocracy":
            self.combiner = MeritocracyCombiner()
        else:
            raise UnknownCombinerException()

    @combination_policy.deleter
    def combination_policy(self):
            del self._combination_policy

    def predict(self, multiplier=1, prediction_filter=None):
        self.word_predictions[:] = []
        self.sent_predictions[:] = []
        self.spell_word_predictions[:] = []
        sent_nextLetterProbs = []
        sent_result = []
        word_result = []
        word_nextLetterProbs = []
        spell_word_result = []
        spell_word_nextLetterProbs = []
        result = (word_nextLetterProbs, word_result, sent_nextLetterProbs, sent_result)

        context = self.context_tracker.token(0)
        # if self.context_tracker:
        #     context = self.context_tracker.past_stream()
        # else:
        #     context = ""

        try:
            for predictor in self.registry:
                
                if(predictor.name == PredictorNames.SentenceComp.value):
                    sentences = predictor.predict(self.max_partial_prediction_size * multiplier, prediction_filter)
                    self.sent_predictions.append(sentences)
                    sent_nextLetterProbs, sent_result = self.combiner.combine(self.sent_predictions, context)

                elif(predictor.name == PredictorNames.CannedPhrases.value):
                    sentences, words = predictor.predict(self.max_partial_prediction_size * multiplier, prediction_filter)
                    sent_result = sentences
                    if(words!=[]):
                        for w in words:
                            self.word_predictions.append(w)

                ### If the predictor is spell predictor, use the predictions only if the other predictors return empty lists
                elif(predictor.name == PredictorNames.Spell.value):
                    self.spell_word_predictions.append(predictor.predict(
                        self.max_partial_prediction_size * multiplier, prediction_filter)
                    )
                    spell_word_nextLetterProbs, spell_word_result = self.combiner.combine(self.spell_word_predictions, context)
                else:
                    self.word_predictions.append(predictor.predict(
                        self.max_partial_prediction_size * multiplier, prediction_filter)
                    )
                    word_nextLetterProbs, word_result = self.combiner.combine(self.word_predictions, context)

            #TODO - CHECK WITH SHACHI ON THIS!!!
            if(predictor.name == PredictorNames.Spell.value and word_result==[]):
                word_result = spell_word_result
                word_nextLetterProbs = spell_word_nextLetterProbs

            result = (word_nextLetterProbs, word_result, sent_nextLetterProbs, sent_result)

        except Exception as e:
            predictor.logger.error(f"Error in PredictorActivator: {e}")
            pass

        finally:
            return result

    def recreate_canned_phrasesDB(self):
        for predictor in self.registry:
            if(predictor.name == PredictorNames.CannedPhrases.value or predictor.name == PredictorNames.CannedWord.value):
                personalized_resources_path = Path(self.config.get(predictor.name, "personalized_resources_path")).as_posix()
                personalized_cannedphrases = os.path.join(personalized_resources_path, self.config.get(predictor.name, "personalized_cannedphrases"))
                pers_cannedphrasesLines = open(personalized_cannedphrases, "r").readlines()
                pers_cannedphrasesLines = [s.strip() for s in pers_cannedphrasesLines]

                predictor.recreate_canned_db(pers_cannedphrasesLines)

    def update_params(self, test_gen_sentence_pred,retrieve_from_AAC):
        for predictor in self.registry:
            predictor.load_model(test_gen_sentence_pred,retrieve_from_AAC)

    def read_updated_toxicWords(self):
        for predictor in self.registry:
            predictor.read_personalized_toxic_words()

    def learn_text(self, text):
        for predictor in self.registry:
            predictor.learn(text)

    # def set_log(self,filename, pathLoc, level):
    #     # provide a way to set the log location for the predictor activator
    #     pass