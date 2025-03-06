# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ConvAssistMessageTypes(IntEnum):
    NOTREADY = 0
    SETPARAM = 1
    NEXTWORDPREDICTION = 2
    NEXTWORDPREDICTIONRESPONSE = 3
    NEXTSENTENCEPREDICTION = 4
    NEXTSENTENCEPREDICTIONRESPONSE = 5
    LEARNWORDS = 6
    LEARNCANNED = 7
    LEARNSHORTHAND = 8
    LEARNSENTENCES = 9
    FORCEQUITAPP = 10
    READYFORPREDICTIONS = 11
    KEYWORDPREDICTIONREQUEST = 12
    KEYWORDRESPONSEREQUEST = 13


class ConvAssistPredictionTypes(IntEnum):
    NONE = 0
    NORMAL = 1
    SHORTHANDMODE = 2
    CANNEDPHRASESMODE = 3
    SENTENCES = 4


class ParameterType(IntEnum):
    NONE = 0
    PATH = 1
    SUGGESTIONS = 2
    TESTGENSENTENCEPRED = 3
    RETRIEVEAAC = 4
    PATHSTATIC = 5
    PATHPERSONALIZED = 6
    PATHLOG = 7
    ENABLELOGS = 8



### "Other: "how are you" \n User:"i am fine" \n
### Need to add keyword ------- DONE
### ConvAssist - Separate sentnece modes. (also modify return format) - TODOOO: Need to return 2 sets of sentences. 
### Number of utterances/window of dialog to be sent from ACAT
### if CRG is off : 
### Parameters to be added for #responses, #sentence completions , # words, #keywords
### Change WordAndCharacterPredictionResponse to PredictionResponse
### if user has typed partial string, and new ASR comes in ... 

### if last is other:
##### Check for the previous user turn. If it is partial, then do not update word prediction... 
##### ignore 

###  ===> CRG responses (keywords n responses - NO WORD prediction)

class ConvAssistMessage:
    MessageType: int
    PredictionType: int
    Data: str
    CRG: bool
    Keyword: str

    @staticmethod
    def jsonDeserialize(obj: Any) -> 'ConvAssistMessage':
        _MessageType = int(obj.get("MessageType"))
        _PredictionType = int(obj.get("PredictionType"))
        _Data = str(obj.get("Data")) 
        _CRG  = bool(obj.get("CRG"))
        _Keyword  = str(obj.get("Keyword"))
        return ConvAssistMessage(_MessageType, _PredictionType, _Data, _CRG, _Keyword)


    def jsonSerialize(self) -> str:
        return json.dumps(dataclasses.asdict(self))
        # Example output: '{"MessageType": 1, "PredictionType": 1, "Data": "test"}'

    def __repr__(self) -> str:
        return (
            f"MessageType: {ConvAssistMessageTypes(self.MessageType).name}, "
            f"PredictionType: {ConvAssistPredictionTypes(self.PredictionType).name}, "
            f"Data: {self.Data},"
            f"CRG: {self.CRG},"
            f"Keyword: {self.Keyword}"
        )


@dataclass
class ConvAssistSetParam:
    Parameter: ParameterType
    Value: Any

    @staticmethod
    def custom_parser(obj):
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    obj[key] = int(value)
                except ValueError:
                    if value.lower() == "true":
                        obj[key] = True
                    elif value.lower() == "false":
                        obj[key] = False
                    else:
                        obj[key] = value
        return obj

    @staticmethod
    def jsonDeserialize(json_str: str) -> "ConvAssistSetParam":
        data = json.loads(json_str, object_hook=ConvAssistSetParam.custom_parser)
        _Parameter = ParameterType(int(data.get("Parameter")))
        _Value = data.get("Value")
        return ConvAssistSetParam(_Parameter, _Value)

    def jsonSerialize(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def __repr__(self) -> str:
        return f"Parameter: {ParameterType(self.Parameter).name}, Value: {self.Value}"


@dataclass
class WordAndCharacterPredictionResponse:
    MessageType: int = ConvAssistMessageTypes.NOTREADY
    PredictionType: int = ConvAssistPredictionTypes.NONE
    PredictedWords: str = ""
    NextCharacters: str = ""
    PredictedKeywords: str = ""
    NextCharactersSentence: str = ""
    PredictedSentence: str = ""
    PredictedKeywordResponse: str = ""

    @staticmethod
    def jsonDeserialize(json_str: str) -> "WordAndCharacterPredictionResponse":
        data = json.loads(json_str)
        return WordAndCharacterPredictionResponse(**data)

    def jsonSerialize(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def __repr__(self) -> str:
        return (
            f"MessageType: {ConvAssistMessageTypes(self.MessageType).name}, "
            f"PredictionType: {ConvAssistPredictionTypes(self.PredictionType).name}, "
            f"PredictedWords: {self.PredictedWords}, "
            f"NextCharacters: {self.NextCharacters}, "
            f"PredictedKeywords: {self.PredictedKeywords},"
            f"NextCharactersSentence: {self.NextCharactersSentence}, "
            f"PredictedSentence: {self.PredictedSentence},"
            f"PredictedKeywordResponse: {self.PredictedKeywordResponse}"
        )
