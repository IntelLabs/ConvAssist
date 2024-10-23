# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ConvAssistMessageTypes(IntEnum):
    NONE = 0
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


@dataclass
class ConvAssistMessage:
    MessageType: int
    PredictionType: int
    Data: str

    @staticmethod
    def jsonDeserialize(obj: Any) -> "ConvAssistMessage":
        _MessageType = int(obj.get("MessageType"))
        _PredictionType = int(obj.get("PredictionType"))
        _Data = str(obj.get("Data"))
        return ConvAssistMessage(_MessageType, _PredictionType, _Data)

    def jsonSerialize(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def __repr__(self) -> str:
        return (
            f"MessageType: {ConvAssistMessageTypes(self.MessageType).name}, "
            f"PredictionType: {ConvAssistPredictionTypes(self.PredictionType).name}, "
            f"Data: {self.Data}"
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
    MessageType: int = ConvAssistMessageTypes.NONE
    PredictionType: int = ConvAssistPredictionTypes.NONE
    PredictedWords: str = ""
    NextCharacters: str = ""
    NextCharactersSentence: str = ""
    PredictedSentence: str = ""

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
            f"NextCharactersSentence: {self.NextCharactersSentence}, "
            f"PredictedSentence: {self.PredictedSentence}"
        )
