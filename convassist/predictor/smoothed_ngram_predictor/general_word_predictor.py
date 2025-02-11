# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import json
import os

from convassist.predictor.smoothed_ngram_predictor.smoothed_ngram_predictor import SmoothedNgramPredictor


class GeneralWordPredictor(SmoothedNgramPredictor):
    def configure(self):
        super().configure()

        # Store the set of most frequent starting words based on an AAC dataset
        # These will be displayed during empty context
        if not os.path.isfile(self.startwords):
            aac_lines = open(self.aac_dataset).readlines()
            startwords = []
            for line in aac_lines:
                w = line.lower().split()[0]
                startwords.append(w)
            counts = collections.Counter(startwords)
            total = sum(counts.values())
            self.precomputed_StartWords = {k: v / total for k, v in counts.items()}
            with open(self.startwords, "w") as fp:
                json.dump(self.precomputed_StartWords, fp)

    def extract_svo(self, sent):
        return sent

    # Override default properties
    @property
    def aac_dataset(self):
        return os.path.join(self._static_resources_path, self._aac_dataset)

    @property
    def database(self):
        return os.path.join(self._static_resources_path, self._database)

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

