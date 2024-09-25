from convassist.ConvAssist import ConvAssist
from convassist.predictor_registry import predictors

conv_assist_modes = {
    "default": {
        "description": "Initial configuration based on ini file.",
        "predictors": [],
    },
    "all": {
        "description": "Predicts the next word and sentence using all predictors.",
        "predictors": [
            "SentenceCompletionPredictor",
            "GeneralWordPredictor",
            "SpellCorrectPredictor",
            "CannedPhrasesPredictor",
            "CannedWordPredictor",
            "ShortHandPredictor",
        ],
    },
    "sentence": {
        "description": "Predicts the next word and sentence.",
        "predictors": [
            "SentenceCompletionPredictor",
            "GeneralWordPredictor",
            "SpellCorrectPredictor",
        ],
    },
    "canned": {
        "description": "Predicts the next word and sentence using canned responses.",
        "predictors": [
            "CannedPhrasesPredictor",
            "CannedWordPredictor",
        ],
    },
    "shorthand": {
        "description": "Predicts the next word using shorthand.",
        "predictors": [
            "ShortHandPredictor",
        ],
    },
}


class ConvAssistMode:
    def __init__(self, convassist: ConvAssist):
        self.convassist = convassist
        self.mode = conv_assist_modes["default"]

    def list_modes(self):
        for mode in conv_assist_modes:
            print(
                f"{mode}: {conv_assist_modes[mode]['description']} {'(active)' if self.mode == conv_assist_modes[mode] else ''}"
            )

    def set_mode(self, mode: str):
        if mode in conv_assist_modes:
            self.convassist.set_predictors(conv_assist_modes[mode]["predictors"])
            self.mode = conv_assist_modes[mode]
        else:
            raise ValueError(f"Invalid mode: {mode}")
