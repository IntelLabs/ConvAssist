from ..utilities.models.predictions import Predictions

class PredictorResponses:
    def __init__(self):
        self.letter_predictions = Predictions("letter_predictions")
        self.word_predictions = Predictions("word_predictions")
        self.sentence_predictions = Predictions("sentence_predictions")
        self.keyword_predictions = Predictions("keyword_predictions")

    def __repr__(self):
        return f"PredictorResponses: {self.letter_predictions}, {self.word_predictions}, {self.sentence_predictions}, {self.keyword_predictions}"
    