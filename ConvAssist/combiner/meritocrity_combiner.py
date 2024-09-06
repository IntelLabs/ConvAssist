# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict
from ConvAssist.combiner import Combiner
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.predictor.utilities.predictor_names import PredictorNames

#TODO - this isn't the best way to combine the probs (from ngram db and deep
# learning based model, just concat m,n predictions and take the top n

class MeritocracyCombiner(Combiner):
    def __init__(self):
        pass

    """
    Computes probabilities for the next letter - for BCI 
    """
    def computeLetterProbs(self, result:Prediction, context:str) -> list[tuple[str, float]]:
        #### compute letter probability
        totalWords = len(result)
        nextLetterProbs: Dict[str, float] = {}

        for each in result:
            word_predicted = each.word.lower().strip()

            nextLetter = " "
            if(each.predictor_name !=PredictorNames.Spell.value):
                if(each.predictor_name == PredictorNames.SentenceComp.value):
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
        result = Prediction()
        for prediction in predictions:
            for suggestion in prediction:
                result.add_suggestion(suggestion)

        nextLetterProb = self.computeLetterProbs(result, context)
        return (nextLetterProb, self.filter(result))
