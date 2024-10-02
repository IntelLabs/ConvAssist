# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict

from ..predictor import SentenceCompletionPredictor, SpellCorrectPredictor
from ..predictor.utilities.prediction import Prediction
from .combiner import Combiner

# TODO - this isn't the best way to combine the probs (from ngram db and deep
# learning based model, just concat m,n predictions and take the top n


class MeritocracyCombiner(Combiner):
    def __init__(self):
        pass

    """
    Computes probabilities for the next letter - for BCI
    """

    def computeLetterProbs(self, result: Prediction, context: str) -> list[tuple[str, float]]:

        # TODO - Check if total words should include empty words
        # Filter out any empty word_predictions from result
        filtered_result = [item for item in result if item.word.lower().strip()]
        totalWords = len(result)
        nextLetterProbs: Dict[str, float] = {}

        for suggestion in filtered_result:
            word_predicted = suggestion.word

            # TODO - refactor so the SpellCorrectPredictor has it's own combiner
            if suggestion.predictor_name == SpellCorrectPredictor.__name__:
                # skip predictions from the SpellCorrectPredictor?!?
                continue

            # TODO - refactor so the SentenceCompletionPredictor has it's own combiner
            elif suggestion.predictor_name == SentenceCompletionPredictor.__name__:
                # result is a sentence. Make sure to get the first letter of the
                # sentence
                nextLetter = word_predicted.split()[0][0]

            elif context:
                position = word_predicted.find(context)
                if (position == 0) and word_predicted != context:
                    nextLetter = word_predicted[position + len(context)]
                else:
                    # if the context is not found in the word_predicted, then
                    # skip this prediction
                    nextLetter = word_predicted[0]
            else:
                nextLetter = word_predicted[0]

            # Now we have the next letter, update the probabilities
            if nextLetter in nextLetterProbs:
                nextLetterProbs[nextLetter] = nextLetterProbs[nextLetter] + 1
            else:
                nextLetterProbs[nextLetter] = 1

        nextLetterProbsList = []
        for k, v in nextLetterProbs.items():
            nextLetterProbsList.append((k, v / totalWords))

        return nextLetterProbsList

    def combine(self, predictions, context):
        result = Prediction()
        for prediction in predictions:
            for suggestion in prediction:
                result.add_suggestion(suggestion)

        nextLetterProb = self.computeLetterProbs(result, context)
        return (nextLetterProb, self.filter(result))
