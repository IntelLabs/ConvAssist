import unittest
import configparser
from unittest.mock import MagicMock, patch

from parameterized import parameterized
from convassist.context_tracker import ContextTracker
from convassist.predictor.contextaware_predictor import ContextAwarePredictor
from transformers import pipeline, AutoTokenizer

class TestCRGPredictor(unittest.TestCase):

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def setUp(self, mock_cuda, mock_mps):
        # setup_utils.setup_static_resources()
        # setup_utils.setup_personalized_resources()

        # SOURCE_DIR = setup_utils.SOURCE_DIR

        self.config = configparser.ConfigParser()
        # self.config["Common"] = {
        #     "static_resources_path": f"{SOURCE_DIR}/test_data/static",
        #     "personalized_resources_path": f"{SOURCE_DIR}/test_data/personalized",
        #     "deltas": "0.01 0.1 0.89",
        #     "stopwords": "stopwords.txt",
        # }

        self.config["PredictorRegistry"] = {
            "predictors": "test_predictor",
        }
        self.config["test_predictor"] = {
            "predictor_class":"CRGPredictor",
            "modelname":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "tokenizer":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "predictiontype":"keywords",
            "prompt":"""You are provided with a dialog between a user and a visitor. 
                The user now needs to respond and could respond in many ways. 
                Generate different possible keywords that the user could use to respond to the visitor's utterance.

                Dialog: {context}
        
                Keywords:
            """
        }

        self.context_tracker = ContextTracker(self.config)

        self.predictor = ContextAwarePredictor(self.config, self.context_tracker, "test_predictor")

    def test_init(self):
        self.assertNotEqual(self.predictor.prompt, "")
        self.assertNotEqual(self.predictor.modelname, "")
        self.assertIsNotNone(self.predictor.generatorPipeline)
        self.assertIsNotNone(self.predictor.tokenizer)
        self.assertTrue(self.predictor.llmLoaded)
        self.assertEqual(self.predictor.predictiontype, "keywords")

    def test_predict_model_not_loaded(self):
        self.predictor.llmLoaded = False
        responses = self.predictor.predict()
        # self.predictor.logger.error.assert_called_with("Model not loaded")
        self.assertEqual(responses.sentence_predictions, [])

    @parameterized.expand(
        [
            ("sentences", "test prompt"),
            ("words", "test prompt"),
            ("keywords", "test prompt")
        ]
    )
    @patch.object(ContextAwarePredictor, '_generate')
    def test_predict(self, predictiontype, prompt, mock_generate):
        self.predictor.llmLoaded = True
        self.predictor.predictiontype = predictiontype
        self.predictor.prompt = prompt
        mock_generate.return_value = []

        responses = self.predictor.predict()
        self.assertEqual(responses.sentence_predictions, [])
        mock_generate.assert_called_once()

if __name__ == "__main__":
    unittest.main()