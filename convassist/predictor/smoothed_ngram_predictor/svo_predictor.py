# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os

from convassist.predictor.smoothed_ngram_predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from convassist.predictor.utilities.svo_util import SVOUtil

class SVOPredictor(SmoothedNgramPredictor):

    def configure(self):
        self.svo_util = SVOUtil(self.stopwordsFile)

        super().configure()


    def extract_svo(self, sent) -> str:
        return self.svo_util.extract_svo(sent)


    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)

