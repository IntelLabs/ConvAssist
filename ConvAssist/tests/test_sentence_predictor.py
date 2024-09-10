from configparser import ConfigParser
import unittest
from unittest.mock import MagicMock, patch
from ConvAssist.predictor.sentence_completion_predictor import SentenceCompletionPredictor

class TestSentenceCompletionPredictor(unittest.TestCase):

    def setUp(self):
        self.context_tracker = MagicMock()
        self.predictor_name = "test_predictor"
        self.config = ConfigParser()
        self.config["Logging"] = {
            "log_location": "",
            "log_level": "INFO"
        }
        self.config["SentenceCompletionPredictor"] = {
            "blacklist_file": "filter_words.txt",
            "test_generalsentenceprediction": "False",
            "static_resources_path": "C:\\Users\\mbeale\\source\\repos\\ConvAssist\\ACAT_ConvAssist_Interface\\ConvAssistCPApp\\resources\\static_resources",
            "personalized_resources_path": "C:\\Users\\mbeale\\source\\repos\\ConvAssist\\ACAT_ConvAssist_Interface\\ConvAssistCPApp\\resources\\personalized",

        }
        self.predictor = SentenceCompletionPredictor(
            config=self.config,
            context_tracker=self.context_tracker,
            predictor_name=self.predictor_name
        )

    def test_read_config(self):

        # Assertions to check if the attributes were set correctly
        self.assertEqual(self.predictor._blacklist_file,"filter_words.txt")
        self.assertEqual(self.predictor._test_generalsentenceprediction, False)

        # Assertions to check if defaults were set correctly
        self.assertEqual(self.predictor._embedding_cache_path, "all_aac_embeddings.pkl")


    @patch('ConvAssist.predictor.sentence_completion_predictor.SentenceCompletionPredictor._read_config')
    def test_read_config_exception(self):
        # Mock the config to raise an exception
        self.config.sections.side_effect = Exception("Config error")

        with self.assertRaises(Exception) as context:
            self.predictor._read_config()

        self.assertTrue("Config error" in str(context.exception))

if __name__ == "__main__":
    unittest.main()