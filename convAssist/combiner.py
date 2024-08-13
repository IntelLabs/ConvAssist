# Copyright (C) 2023 Intel Corporation


# SPDX-License-Identifier: Apache-2.0


"""
Combiner classes to merge results from several predictors.

"""
import abc

import convAssist.word_sentence_predictor

from convAssist.word_sentence_predictor import * 

from convAssist.logger import ConvAssistLogger
convAssistLog = ConvAssistLogger().configure(True, ConvAssistLogger.INFO, "ConvAssist_Predictor.log")
convAssistLog.info("LOGGING IN COMBINER")
class Combiner(object):
    """
    Base class for all combiners
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def filter(self, prediction):
        seen_tokens = set()
        result = convAssist.word_sentence_predictor.Prediction()
        for i, suggestion in enumerate(prediction):
            token = suggestion.word
            if token not in seen_tokens:
                for j in range(i + 1, len(prediction)):
                    if token == prediction[j].word:
                        # TODO: interpolate here?
                        suggestion.probability += prediction[j].probability
                        if suggestion.probability > convAssist.word_sentence_predictor.MAX_PROBABILITY:
                            suggestion.probability = convAssist.MAX_PROBABILITY
                seen_tokens.add(token)
                result.add_suggestion(suggestion)
        return result

    @abc.abstractmethod
    def combine(self):
        raise NotImplementedError("Method must be implemented")

#this isn't the best way to combine the probs (from ngram db and deep learning based model, TODO - just concat m,n predictions
class MeritocracyCombiner(Combiner):
    def __init__(self):
        pass
    """
    Computes probabilities for the next letter - for BCI 

    """
    def computeLetterProbs(self, result, context):
        #### compute letter probability
        totalWords = len(result)
        nextLetterProbs = {}
        for each in result:
            word_predicted = each.word.lower().strip()
            convAssistLog.debug(f"computeLetterProbs: context = {context} word_predicted = {word_predicted} predictor = {each.predictor_name}")
            nextLetter = " "
            if(each.predictor_name !=convAssist.word_sentence_predictor.PredictorNames.Spell.value):
                if(each.predictor_name == convAssist.word_sentence_predictor.PredictorNames.SentenceComp.value):
                    if(word_predicted!=""):     
                        nextLetter = word_predicted.strip().split()[0][0]

                else:
                    ####### check to ensure there is some word_prediction         
                    if(word_predicted!=""):
                        ####### if context is not empty, split the word_predicted to compute the next letter
                        if(context!="" and context!=" "):
                            if(word_predicted != context):
                                nextLetter = word_predicted[len(context):][0]
                                # print("word_pred = ", word_predicted, " nextLetter = ", nextLetter)
                        else:
                            ####### if context is empty, pick the first letter of the word_predicted as the next letter
                            nextLetter = word_predicted[0]

            if (nextLetter in nextLetterProbs):
                nextLetterProbs[nextLetter] = nextLetterProbs[nextLetter] + 1
            else:
                nextLetterProbs[nextLetter] = 1
        nextLetterProbsList = []
        for k, v in nextLetterProbs.items():
            nextLetterProbsList.append((k,v / totalWords))
        return nextLetterProbsList

    def combine(self, predictions, context):
        result = convAssist.word_sentence_predictor.Prediction()
        for prediction in predictions:
            for suggestion in prediction:
                result.add_suggestion(suggestion)

        convAssistLog.debug(f"Combiner.combine: Result {result}")
        nextLetterProb = self.computeLetterProbs(result, context)
        return (nextLetterProb, self.filter(result))
