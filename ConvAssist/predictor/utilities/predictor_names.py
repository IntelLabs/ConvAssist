# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

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
