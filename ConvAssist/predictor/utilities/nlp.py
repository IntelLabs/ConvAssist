# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

from spacy import load

from ConvAssist.utilities.singleton import Singleton


class NLP(metaclass=Singleton):
    def __init__(self):
        self.nlp = self.load_nlp()

    def load_nlp(self):
        nlp_loc = "en_core_web_sm"
        # spacy model is in _MEIPASS when running as a pyinstaller executable
        if hasattr(sys, "_MEIPASS"):
            base_path = sys._MEIPASS  # type: ignore
            nlp_loc = os.path.join(base_path, nlp_loc)

        nlp = load(nlp_loc)

        return nlp

    def get_nlp(self):
        return self.nlp
