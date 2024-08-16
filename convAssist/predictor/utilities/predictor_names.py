# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

class PredictorNames(Enum):
    """
    Define names of all predictors
    """
    SentenceComp = "SentenceCompletionPredictor"
    CannedWord = "CannedWordPredictor"
    GeneralWord = "DefaultSmoothedNgramPredictor"
    PersonalizedWord = "PersonalSmoothedNgramPredictor"
    Spell = "SpellCorrectPredictor"
    ShortHand = "ShortHandPredictor"
    CannedPhrases = "CannedPhrasesPredictor"