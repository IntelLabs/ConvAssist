from .predictor import Predictor
from .utilities import Predictions, Suggestion, PredictorResponses
import transformers
from transformers import AutoTokenizer

class CRGPredictor(Predictor):
    @property
    def modelname(self):
        return self._modelname

    @modelname.setter
    def modelname(self, value):
        self._modelname = value

    @property
    def generatorPipeline(self):
        return self._generatorPipeline
    
    @generatorPipeline.setter
    def generatorPipeline(self, value):
        self._generatorPipeline = value

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def llmLoaded(self):
        return self._llmLoaded
    
    @llmLoaded.setter
    def llmLoaded(self, value):
        self._llmLoaded = value

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

    def __init__(self, *args, **kwargs):
        self._prompt: str = ""
        self._modelname: str = ""
        self._device: str = ""
        self._n_gpu: int = 0
        self._generatorPipeline: transformers.pipelines.Pipeline = None
        self._tokenizer: AutoTokenizer = None
        self._llmLoaded: bool = False
        self._predictiontype: str = ""

        super().__init__(*args, **kwargs)
        
    def configure(self) -> None:

        try:
            self.generatorPipeline = transformers.pipeline("text-generation", model=self.modelname, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

            self.llmLoaded = True
            self.logger.info(f"Model {self.modelname} loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model {self.modelname}: {e}")

    def predict(self, max_partial_prediction_size=None, filter=None) -> PredictorResponses:
        responses:PredictorResponses = PredictorResponses()

        match self.predictiontype:
            case "sentences":
                predictions = responses.sentence_predictions
            case "words":
                predictions = responses.word_predictions
            case "keywords":
                predictions = responses.keyword_predictions
            case _:
                raise ValueError(f"Invalid prediction type: {self.predictiontype}")

        if not self.llmLoaded:
            self.logger.error("Model not loaded")
            return responses
        
        context = self.context_tracker.context.lstrip()

        predictions = self._generate(self.prompt, context, predictions)

        return responses
    
    def _generate(self, prompt, context: str, predictions: Predictions) -> Predictions:
        prompt = self.prompt.replace("{context}", context).replace("{keywords}", "Yes, No, Maybe")

        self.logger.debug(f"Prompt: {prompt}")
        self.logger.debug(f"Context: {context}")

        try:
            outputs = self.generatorPipeline(
                prompt,
                do_sample=False,
                num_return_sequences=5,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=1.5,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=100,
                return_full_text=False,
            )
 

        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return []
                
        for index, generated_text in enumerate(outputs):
            sentence_text = generated_text['generated_text']
            predictions.add_suggestion(Suggestion(sentence_text, 1/len(outputs), self.predictor_name))

        return predictions
