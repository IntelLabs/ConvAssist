# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
from pathlib import Path

import spacy

from ...utilities.singleton import Singleton


class NLP(metaclass=Singleton):
    def __init__(self):
        self.nlp = self.load_nlp()

    def load_nlp(self):
        nlp_loc = "en_core_web_sm"
        # spacy model is in _MEIPASS when running as a pyinstaller executable
        if hasattr(sys, "_MEIPASS"):  # pragma: no cover
            base_path = sys._MEIPASS  # type: ignore
            nlp_loc = os.path.join(base_path, nlp_loc)
            print("nlp_loc:", nlp_loc)

        if os.path.exists(nlp_loc):
            # Loading the model from a path
            child_dirs = [child for child in Path(nlp_loc).iterdir() if child.is_dir()]
            if len(child_dirs) > 0:
                nlp = spacy.load(child_dirs[0])

        else:
            # Loading the model from the installed package
            if not spacy.util.is_package(nlp_loc):
                spacy.cli.download(nlp_loc)

            nlp = spacy.load(nlp_loc)

        return nlp

    def get_nlp(self):
        return self.nlp
