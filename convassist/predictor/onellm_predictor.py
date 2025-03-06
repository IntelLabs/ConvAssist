from .predictor import Predictor
from .utilities.prediction import Prediction
import transformers
import torch
from transformers import AutoTokenizer
import os
from pathlib import Path

class OneLLMPredictor(Predictor):
    model = None
    tokenizer = None
    llmLoaded = False
    generatorPipeline = None

    def __init__(self, config, context_tracker, predictor_name, logger=None):
        super().__init__(config, context_tracker, predictor_name, logger)
        self.logger.info(f"Initializing {self.predictor_name} predictor")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        if not OneLLMPredictor.llmLoaded:
            self.load_oneLLM()

    def load_oneLLM(self):
        for predictor in self.registry:
            if OneLLMPredictor.llmLoaded:
                return OneLLMPredictor.llmLoaded
            static_resources_path = Path(self.config.get(predictor.name, "static_resources_path"))
            modelname = os.path.join(static_resources_path, self.config.get(predictor.name, "oneLLM_model"))
            self.logger.debug(f"Loading model {modelname}")

            try:
                OneLLMPredictor.generatorPipeline = transformers.pipeline("text-generation", model="Qwen/Qwen1.5-4B-Chat")
                OneLLMPredictor.tokenizer = AutoTokenizer.from_pretrained(modelname)
                OneLLMPredictor.llmLoaded = True
                self.logger.info(f"Model {modelname} loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model {modelname}: {e}")

        return OneLLMPredictor.llmLoaded

    def configure(self) -> None:
        self.logger.info(f"Configuring {self.predictor_name} predictor")

    def predict(self, max_partial_prediction_size=None, filter=None) -> tuple[Prediction, Prediction]:
        raise NotImplementedError("This method should be implemented in subclasses")