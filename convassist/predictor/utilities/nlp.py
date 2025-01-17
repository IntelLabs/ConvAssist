# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import spacy

# from spacy import load, util, cli
from spacy.language import Language

from ...utilities.singleton import Singleton


class NLP(metaclass=Singleton):
    def __init__(self):
        self._nlp: Language = None

    def get_nlp(self, package_name="en_core_web_sm") -> Language:
        if self._nlp is None:
            spacy.cli.download(package_name)
            self._nlp = spacy.load(package_name)

        return self._nlp
