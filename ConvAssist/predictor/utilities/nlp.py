# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import spacy

from ConvAssist.utilities.singleton import Singleton

class NLP(metaclass=Singleton):
    def __init__(self):
        self.nlp = self.load_nlp()
    
    def load_nlp(self):
        try:
            base_path = sys._MEIPASS
            en_core_path = base_path+"/en_core_web_sm/"
            files = os.listdir(en_core_path)
            for file in files:
                if file.lower().startswith("en_core_web_sm"):
                    file_path = os.path.join(en_core_path, file)
                    nlp = spacy.load(file_path)

        except Exception as e:
            nlp = spacy.load("en_core_web_sm")
        
        return nlp
    
    def get_nlp(self):
        return self.nlp