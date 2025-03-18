from __future__ import annotations
from ..utilities.models.predictions import Predictions
from convassist.combiner.meritocrity_combiner import MeritocracyCombiner

class PredictorResponse:
    def __init__(self):
        self.NextWordCharacters = Predictions("next_word_chars")
        self.wordPredictions = Predictions("wordPredictions")
        self.NextSentenceCharacters = Predictions("next_sentence_chars")
        self.sentencePredictions = Predictions("sentencePredictions")
        self.NextKeywordCharacters = Predictions("next_keyword_chars")
        self.keywordPredictions = Predictions("keywordPredictions")

        self.combiner = MeritocracyCombiner()

    def __repr__(self):
         return f"PredictorResponses: Words: {len(self.wordPredictions)} " \
                f"Sentences: {len(self.sentencePredictions)} " \
                f"Keywords: {len(self.keywordPredictions)}"
    
    def extend(self, predictor_response):
        if not isinstance(predictor_response, PredictorResponse):
            raise ValueError("Cannot extend with a non-PredictorResponse object")
        
        self.wordPredictions.extend(predictor_response.wordPredictions)
        self.sentencePredictions.extend(predictor_response.sentencePredictions)
        self.keywordPredictions.extend(predictor_response.keywordPredictions)

    def combine(self, context) -> PredictorResponse:
        # for each response type, combine the predictions and generate the next letter probabilities
        
        #word predictions
        self.NextWordCharacters, self.wordPredictions = self.combiner.combine(
            self.wordPredictions, context
        )

        #sentence predictions
        self.NextSentenceCharacters, self.sentencePredictions = self.combiner.combine(
            self.sentencePredictions, context
        )

        #keyword predictions
        self.keywordPredictions = self.combiner.filter(self.keywordPredictions)
        # self.NextKeywordCharacters, self.keywordPredictions = self.combiner.combine(
        #     self.keywordPredictions, context
        # )

        return self
