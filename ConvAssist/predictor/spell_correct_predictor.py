# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from configparser import ConfigParser
import logging
import re
import os
from collections import Counter

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.predictor import Predictor
from ConvAssist.predictor.utilities.suggestion import Suggestion
from ConvAssist.predictor.utilities.prediction import Prediction


class SpellCorrectPredictor(Predictor):
    """Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

    Copyright (c) 2007-2016 Peter Norvig
    MIT license: www.opensource.org/licenses/mit-license.php

    """

    def __init__(
            self,
            config: ConfigParser,
            context_tracker: ContextTracker,
            predictor_name: str,
            logger: logging.Logger | None = None
    ):
        super().__init__(
            config,
            context_tracker,
            predictor_name,
            logger
        )

        if os.path.exists(self.spellingdatabase):
            with open(self.spellingdatabase) as file:
                self.WORDS = Counter(self._words(file.read()))
        else:
            self.WORDS = Counter()

    @property
    def spellingdatabase(self):
        return os.path.join(self._static_resources_path, self._spellingdatabase)

    def _words(self, text): return re.findall(r'\w+', text.lower())

    def _P(self, word):
        """ Probability of `word`."""
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N

    def _correction(self, word):
        """Most probable spelling correction for word."""
        return max(self._candidates(word), key=self._P)

    def _candidates(self, word):
        """Generate possible spelling corrections for word."""
        return (self._known([word]) or self._known(self._edits1(word)) or self._known(self._edits2(word)) or [word])

    def _known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def _edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def predict(self, max_partial_prediction_size = None, filter = None):
        super().predict(max_partial_prediction_size, filter)

        token = self.context_tracker.token(0)
        setence_prediction = Prediction()
        word_prediction = Prediction()

        prefix_completion_candidates = self._candidates(token)

        for j, candidate in enumerate(prefix_completion_candidates):
            probability = self._P(candidate)
            if probability > 0.0001:
                word_prediction.add_suggestion(
                    Suggestion(candidate, probability, self.predictor_name)
                )

        if len(word_prediction) == 0:
            self.logger.error(f"No predictions from SpellCorrectPredictor")

        return setence_prediction, word_prediction
