from .predictor import Predictor
from .utilities import PredictorResponse
from .utilities.models import Predictions
import os
from openai import AzureOpenAI

class AzureOpenAIPredictor(Predictor):

    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    @property
    def predictiontype(self):
        return self._predictiontype
    
    @predictiontype.setter
    def predictiontype(self, value):
        self._predictiontype = value

    @property
    def api_key(self):
        return self._api_key
    
    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def api_version(self):
        return self._api_version
    
    @api_version.setter
    def api_version(self, value):
        self._api_version = value

    @property
    def api_endpoint(self):
        return self._api_endpoint
    
    @api_endpoint.setter
    def api_endpoint(self, value):
        self._api_endpoint = value


    def __init__(self, *args, **kwargs):
        self._prompt: str = ""
        self._predictiontype: str = ""
        self._api_key: str = ""
        self._api_version: str = ""
        self._api_endpoint: str = ""

        super().__init__(*args, **kwargs)
        
    def configure(self) -> None:

        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_endpoint,
            )
            self.logger.info(f"AzureOpenAI loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model {self.modelname}: {e}")

    def predict(self, max_partial_prediction_size=None, filter=None) -> PredictorResponse:
        responses:PredictorResponse = PredictorResponse()

        match self.predictiontype:
            case "sentences":
                predictions = responses.sentencePredictions
            case "words":
                predictions = responses.wordPredictions
            case "keywords":
                predictions = responses.keywordPredictions
            case _:
                raise ValueError(f"Invalid prediction type: {self.predictiontype}")
        
        context = self.context_tracker.context.lstrip()

        predictions = self._generate(self.prompt, context, predictions)

        return responses
    
    def _generate(self, prompt, context: str, predictions: Predictions) -> Predictions:
        self.prompt = self.prompt.replace("context", context)

        self.logger.debug(f"Prompt: {prompt}")
        self.logger.debug(f"Context: {context}")

        try:
            deploymentname = "gpt-35-turbo-instruct"

            response = self.client.completions.create(
                model=deploymentname,
                prompt=self.prompt,
                temperature=1,  
                max_tokens=361,  
                top_p=0.5,  
                frequency_penalty=0.75,  
                presence_penalty=0.5,  
                best_of=5,  
                stop=None  
            )

            print (response.choices[0].text)

        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return []
                
        # for index, generated_text in enumerate(response):
        #     sentence_text = generated_text['generated_text']
        #     predictions.add_suggestion(Suggestion(sentence_text, 1/len(outputs), self.predictor_name))

        return predictions

    
