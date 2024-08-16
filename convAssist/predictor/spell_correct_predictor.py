# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import os
from collections import Counter
from pathlib import Path

from ConvAssist.predictor import Predictor
from ConvAssist.utilities.suggestion import Suggestion
from ConvAssist.predictor.utilities.prediction import Prediction


class SpellCorrectPredictor(Predictor):
    """Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

    Copyright (c) 2007-2016 Peter Norvig
    MIT license: www.opensource.org/licenses/mit-license.php

    """

    def __init__(
            self,
            config,
            context_tracker,
            predictor_name,
            short_desc=None,
            long_desc=None,
            dbconnection=None,
    ):
        Predictor.__init__(
            self, 
            config, 
            context_tracker, 
            predictor_name, 
            short_desc, 
            long_desc,
            dbconnection
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = None
        self.learn_mode_set = False

        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None

        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()

        if os.path.exists(self.spellingDatabase):
            with open(self.spellingDatabase) as file:
                self.WORDS = Counter(self.words(file.read()))
        else:
            self.WORDS = Counter()

    def words(self, text): return re.findall(r'\w+', text.lower())

    def P(self, word):
        """ Probability of `word`."""
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self, word):
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def predict(self, max_partial_prediction_size, filter):
        token = self.context_tracker.token(0)
        prediction = Prediction()
        prefix_completion_candidates = self.candidates(token)

        for j, candidate in enumerate(prefix_completion_candidates):
            probability = self.P(candidate)
            if probability > 0.0001:
                prediction.add_suggestion(
                    Suggestion(candidate, probability, self.name)
                )
        return prediction

    def learn(self, text):
        pass

    def _read_config(self):
        self.static_resources_path = Path(self.config.get(self.name, "static_resources_path"))
        self.spellingDatabase = os.path.join(self.static_resources_path, self.config.get(self.name, "spellingDatabase"))