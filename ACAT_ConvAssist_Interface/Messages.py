"""
 Copyright (C) 2023 Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""

"""
Data classes for serializable json strings
"""
"""
Imports
"""
from typing import Any
from dataclasses import dataclass
version_ConvAssist = "V 1.0.0"

@dataclass
class ConvAssistMessage:
    MessageType: int
    PredictionType: int
    Data: str

    @staticmethod
    def jsonDeserialize(obj: Any) -> 'ConvAssistMessage':
        _MessageType = int(obj.get("MessageType"))
        _PredictionType = int(obj.get("PredictionType"))
        _Data = str(obj.get("Data"))
        return ConvAssistMessage(_MessageType, _PredictionType, _Data)


@dataclass
class ConvAssistSetParam:
    Parameter: int
    Value: str

    @staticmethod
    def jsonDeserialize(obj: Any) -> 'ConvAssistSetParam':
        _Parameter = int(obj.get("Parameter"))
        _Value = str(obj.get("Value"))
        return ConvAssistSetParam(_Parameter, _Value)


@dataclass
class WordAndCharacterPredictionResponse:
    MessageType: int
    PredictionType: int
    PredictedWords: str
    NextCharacters: str
    NextCharactersSentence: str
    PredictedSentence: str

    @staticmethod
    def jsonDeserialize(obj: Any) -> 'WordAndCharacterPredictionResponse':
        _MessageType = int(obj.get("MessageType"))
        _PredictionType = int(obj.get("PredictionType"))
        _PredictedWords = str(obj.get("PredictedWords"))
        _NextCharacters = str(obj.get("NextCharacters"))
        _NextCharactersSentence = str(obj.get("NextCharactersSentence"))
        _PredictedSentence = str(obj.get("PredictedSentence"))
        return WordAndCharacterPredictionResponse(_MessageType, _PredictionType,
                                                  _PredictedWords, _NextCharacters,
                                                  _NextCharactersSentence, _PredictedSentence)
licesnce_text_string2 = "Copyright (c) 2013-2017 Intel Corporation\n" \
                "Licensed under the Apache License, Version 2.0 (the License);\n" \
                "you may not use this file except in compliance with the License.\n" \
                "You may obtain a copy of the License at\n" \
                "->  http://www.apache.org/licenses/LICENSE-2.0\n" \
                "Unless required by applicable law or agreed to in writing, software\n" \
                "distributed under the License is distributed on an AS IS BASIS,\n" \
                "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" \
                "See the License for the specific language governing permissions and\n" \
                "limitations under the License.\n" \
                "ConvAssist is built on Pressagio, that is a library that predicts text \n" \
                "based on n-gram models (https://pressagio.readthedocs.io, https://github.com/Poio-NLP/pressagio).  \n "\
                "Pressagio is a pure Python port of the presage library: https://presage.sourceforge.io \n "\
                "and is part of the Poio project: https://www.poio.eu. \n"\
                " \n" \
                + version_ConvAssist


licesnce_text_string = "Presage, an extensible predictive text entry system\n" \
                "Copyright (C) 2008  Matteo Vescovi <matteo.vescovi@yahoo.co.uk>\n" \
                "This program is free software; you can redistribute it and/or modify\n" \
                "it under the terms of the GNU General Public License as published by\n" \
                "the Free Software Foundation; either version 2 of the License, or\n" \
                "(at your option) any later version.\n" \
                "This program is distributed in the hope that it will be useful,\n" \
                "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" \
                "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" \
                "GNU General Public License for more details.\n" \
                "You should have received a copy of the GNU General Public License along\n" \
                "with this program; if not, write to the Free Software Foundation, Inc.,\n" \
                "51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.\n" \
                " \n" \
                "v 1.0.0.A"


# Example Usage
# jsonstring = json.loads(myjsonstring)
# object = dataclassName.staticmethodName(jsonstring)
