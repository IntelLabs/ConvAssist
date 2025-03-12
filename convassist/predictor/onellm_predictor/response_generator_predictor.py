from .onellm_predictor import OneLLMPredictor
from ..utilities.prediction import Prediction
from ..utilities.suggestion import Suggestion

class KeywordResponseGeneratorPredictor(OneLLMPredictor):
    def __init__(self, config, context_tracker, predictor_name, logger=None):
        print("Inside keywordResponseGeneratorPredictor init")
        super().__init__(config, context_tracker, predictor_name, logger)
        

    def predict(self, max_partial_prediction_size=None, filter=None) -> Prediction:
        self.logger.info(f"Generating sentence with {self.predictor_name} predictor")
        context = self.context_tracker.context.lstrip()
        dialog = context.split("<keyword>")[0]
        keyword = context.split("<keyword>")[1]
        ### TODO: Add prompts to config 
        prompt = """You are provided with a dialog between a user and a visitor after the 'Dialog:' tag. The user has chosen a keyword to respond that is provided after the 'Keyword:' tag. 
        Use the keyword to respond to the dialog. Generate multiple diverse responses. \n\n
        Dialog: {dialog}\n\n
        Keyword: {keyword}\n\n
        Response: \n"""
        sentences = self._generate_sentence(prompt, context)
        onellm_sentence_prediction = Prediction()
        for sentence in sentences:
            sentence = sentence.strip()
            onellm_sentence_prediction.add_suggestion(Suggestion(sentence, 1.0/len(sentences), self.name))
            print("Sentence output from SentencePEdictor_onellm = ", sentence)
        return onellm_sentence_prediction

    def _generate_sentence(self,prompt, context: str) -> list[str]:
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        outputs = self.generatorPipeline(prompt+"\n"+context.strip(),do_sample=False,num_return_sequences=5,num_beams=5, num_beam_groups=5,diversity_penalty=1.5,eos_token_id=self.tokenizer.eos_token_id,max_length=15,)
        # sentence_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sentences = []
        for generated_text in outputs:
            sentence_text = generated_text['generated_text']
            generated_sentences.append(sentence_text)
            self.logger.debug("Prompt:"+prompt+"\n"+"Context:"+context+"\nResponse:"+sentence_text+"\n------\n")
        return generated_sentences
    