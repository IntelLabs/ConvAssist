import pytest
from unittest.mock import MagicMock
from convassist.predictor.utilities import PredictorResponse
from convassist.predictor.utilities.models import Predictions
from convassist.combiner.meritocrity_combiner import MeritocracyCombiner

def test_predictor_response_init():
    response = PredictorResponse()
    
    assert isinstance(response.nextWordCharacters, Predictions)
    assert isinstance(response.wordPredictions, Predictions)
    assert isinstance(response.nextSentenceCharacters, Predictions)
    assert isinstance(response.sentencePredictions, Predictions)
    assert isinstance(response.nextKeywordCharacters, Predictions)
    assert isinstance(response.keywordPredictions, Predictions)

def test_predictor_response_repr():
    response = PredictorResponse()
    repr_str = repr(response)
    
    assert "PredictorResponses: Words:" in repr_str
    assert "Sentences:" in repr_str
    assert "Keywords:" in repr_str

def test_predictor_response_extend():
    response1 = PredictorResponse()
    response2 = PredictorResponse()
    
    response1.wordPredictions.extend = MagicMock()
    response1.sentencePredictions.extend = MagicMock()
    response1.keywordPredictions.extend = MagicMock()
    
    response1.extend(response2)
    
    response1.wordPredictions.extend.assert_called_once_with(response2.wordPredictions)
    response1.sentencePredictions.extend.assert_called_once_with(response2.sentencePredictions)
    response1.keywordPredictions.extend.assert_called_once_with(response2.keywordPredictions)

def test_predictor_response_extend_invalid():
    response = PredictorResponse()
    
    with pytest.raises(ValueError):
        response.extend("invalid_object")

def test_predictor_response_combine():
    response = PredictorResponse()
    context = MagicMock()
    
    response.combiner.combine = MagicMock(return_value=(MagicMock(), MagicMock()))
    response.combiner.filter = MagicMock(return_value=MagicMock())
    
    combined_response = response.combine(context)
    
    assert combined_response is response
