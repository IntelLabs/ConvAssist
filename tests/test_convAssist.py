import unittest
from unittest.mock import MagicMock
from configparser import ConfigParser

from ConvAssist.predictior_activator import PredictorActivator
from ConvAssist.predictor_registry import PredictorRegistry
from ConvAssist.context_tracker import ContextTracker
from ConvAssist.utilities.callback import Callback
from ConvAssist import ConvAssist


class TestConvAssist(unittest.TestCase):
    def setUp(self):
        self.config = ConfigParser()
        self.config["Logging"] = {
            "log_location": "",
            "log_level": "INFO"
        }
        self.callback = Callback()
        
    def test_init(self):
        conv_assist = ConvAssist(self.callback, self.config)

        self.assertEqual(conv_assist.callback, self.callback)
        self.assertEqual(conv_assist.config, self.config)
        self.assertIsInstance(conv_assist.predictor_registry, PredictorRegistry)
        self.assertIsInstance(conv_assist.context_tracker, ContextTracker)
        self.assertIsInstance(conv_assist.predictor_activator, PredictorActivator)
        self.assertEqual(conv_assist.predictor_activator.combination_policy, "meritocracy")

    def test_predict(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the predict method of predictor_activator
        conv_assist.predictor_activator.predict = MagicMock(return_value=(1.0, [], 0.5, []))

        wordprob, word, sentprob, sent = conv_assist.predict()

        self.assertEqual(wordprob, 1.0)
        self.assertEqual(word, [])
        self.assertEqual(sentprob, 0.5)
        self.assertEqual(sent, [])

    def test_update_params(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the update_params method of predictor_activator
        conv_assist.predictor_activator.update_params = MagicMock()

        conv_assist.update_params(True, False)

        conv_assist.predictor_activator.update_params.assert_called_with(True, False)

    def test_read_updated_toxicWords(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the read_updated_toxicWords method of predictor_activator
        conv_assist.predictor_activator.read_updated_toxicWords = MagicMock()

        conv_assist.read_updated_toxicWords()

        conv_assist.predictor_activator.read_updated_toxicWords.assert_called()

    # def test_setLogLocation(self):
    #     conv_assist = ConvAssist(self.callback, self.config)

    #     # Mock the set_log method of predictor_activator
    #     conv_assist.predictor_activator.set_log = MagicMock()

    #     conv_assist.setLogLocation("filename", "pathLoc", "level")

    #     conv_assist.predictor_activator.set_log.assert_called_with("filename", "pathLoc", "level")

    def test_cannedPhrase_recreateDB(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the recreate_canned_phrasesDB method of predictor_activator
        conv_assist.predictor_activator.recreate_canned_phrasesDB = MagicMock()

        conv_assist.cannedPhrase_recreateDB()

        conv_assist.predictor_activator.recreate_canned_phrasesDB.assert_called()

    def test_learn_db(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the learn_text method of predictor_activator
        conv_assist.predictor_activator.learn_text = MagicMock()

        conv_assist.learn_db("This is a test sentence.")

        conv_assist.predictor_activator.learn_text.assert_called_with("This is a test sentence.")

    def test_check_model(self):
        conv_assist = ConvAssist(self.callback, self.config)

        # Mock the model_status method of predictor_registry
        mock_model_status = MagicMock(return_value=1)
        conv_assist.predictor_registry.model_status = mock_model_status

        status = conv_assist.check_model()

        self.assertEqual(status, 1) 

    # def test_close_database(self):
    #     conv_assist = ConvAssist(self.callback, self.config)

    #     # Mock the close_database method of predictor_registry
    #     conv_assist.predictor_registry.close_database = MagicMock()

    #     conv_assist.close_database()

    #     conv_assist.predictor_registry.close_database.assert_called()


if __name__ == "__main__":
    unittest.main()