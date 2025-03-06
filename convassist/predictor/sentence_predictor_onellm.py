from .onellm_predictor import OneLLMPredictor
from .utilities.prediction import Prediction
from .utilities.suggestion import Suggestion

class SentencePredictorOneLLM(OneLLMPredictor):
    def __init__(self, config, context_tracker, predictor_name, logger=None):
        super().__init__(config, context_tracker, predictor_name, logger)

    def predict(self, max_partial_prediction_size=None, filter=None) -> Prediction:
        self.logger.info(f"Generating sentence with {self.predictor_name} predictor")
        context = self.context_tracker.context.lstrip()
        ### Check if user has typed any text
        if(context.endswith("User: ")):
            prompt = "You are given a dialog between a user and a visitor. On behalf of the user, generate a response to the dialog that the user can choose from."
        elif(context.contains("User:")):
            prompt = "You are given a dialog between a user and a visitor. Complete the user's response to the dialog."
        else:
            prompt = "Complete the user's sentence."
        sentences = self._generate_sentence(prompt, context)
        onellm_sentence_prediction = Prediction()
        for sentence in sentences:
            sentence = sentence.strip()
            onellm_sentence_prediction.add_suggestion(Suggestion(sentence, 1.0/len(sentences), self.name))
            print("Sentence output from SentencePEdictor_onellm = ", sentence)
        return  onellm_sentence_prediction

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
    
    # def generate(self, dialog, prompt):
    #     global generatorPipeline
    #     global oneLLMTokenizer
    #     print("prmopt+dialog = ", prompt+dialog)
    #     out = generatorPipeline(prompt+"\n"+dialog.strip(),do_sample=False,num_return_sequences=5,num_beams=5, num_beam_groups=5,diversity_penalty=1.5,eos_token_id=oneLLMTokenizer.eos_token_id,max_length=15,)
    #     text_all = []
    #     for each in out:
    #         toappend = each["generated_text"][len(prompt+dialog):].split(".")[0]
    #         if len(toappend)>0:
    #             toappend = each["generated_text"][len(prompt+dialog):].split("?")[0]
    #         text_all.append(toappend+"\n")
    #     self.logger.debug()
    #     # for each in out:
    #     #     toappend = each["generated_text"][len(prompt+dialog):]
    #     #     toappend = toappend.split("Visitor")[0]
    #     #     text_all.append(toappend+"\n")

    #     convAssistLog.Log("Prompt:"+prompt+"\n"+"Context:"+dialog+"\nResponse1:"+text_all[0]+"\nResponse2:"+text_all[1]+"\nResponse3:"+text_all[2]+"\nResponse4:"+text_all[3]+"\nResponse5:"+text_all[4]+"\n-------------\n")
    #     return (text_all[0:5])



    # def predict(self, max_partial_prediction_size, filter):
    #     global llmLoaded
        
    #     print("inside keyword predictor predict")
    #     tokens = [""] * self.cardinality
    #     prediction = Prediction()
    #     if llmLoaded==False:
    #         return prediction
        

    #     #### input context:  history = messageReceived.Data
    #     context = self.context_tracker.past_stream()
    #     dialog = context
    #     print("INCOMING CONTEXT IN SENTENCE PREDICTOR: \n", dialog)
    #     prompt = "Complete the sentence:"

    #     return_text = self.generate(dialog, prompt)
    #     for each in return_text:
    #         each = each.strip()
    #         prediction.add_suggestion(Suggestion(each, 1.0/len(return_text), self.name))
    #         print(" each output in SentencePEdictor = ", each)
    #     print("prediction = ", prediction)
    #     return prediction

