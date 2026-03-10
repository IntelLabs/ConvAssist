# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import spacy
from spacy.cli.download import download
from convassist.utilities.singleton import Singleton


class NLP(metaclass=Singleton):
    def __init__(self, path: str):
        self.path = path
        self.nlp = self.load_nlp()

    def load_nlp(self):
        nlp_model = "en_core_web_sm"

        try:
            if not spacy.util.is_package(nlp_model):
                download(nlp_model)
            nlp = spacy.load(nlp_model)
            return nlp
        except Exception as e:
            raise RuntimeError(f"Failed to load spaCy model '{nlp_model}': {e}")

    def get_nlp(self):
        return self.nlp
