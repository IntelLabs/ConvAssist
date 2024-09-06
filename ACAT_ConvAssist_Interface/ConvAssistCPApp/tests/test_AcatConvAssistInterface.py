import logging
import unittest
import os
from unittest.mock import MagicMock, patch
from ACAT_ConvAssist_Interface.ConvAssistCPApp.ACATConvAssistInterface import ACATConvAssistInterface

current_dir = os.path.dirname(os.path.realpath(__file__))

class TestACATConvAssistInterface(unittest.TestCase):
    def setUp(self):
        app_quit_event = MagicMock()
        self.interface = ACATConvAssistInterface(app_quit_event)

    def test_createPredictorConfig(self):
        config_file_name = "test.ini"
        self.interface.path = os.path.join(current_dir, "/path/to/files")
        self.interface.pathstatic = os.path.join(current_dir, "/path/to/static")
        self.interface.pathpersonalized = os.path.join(current_dir, "/path/to/personalized")
        self.interface.suggestions = 5
        self.interface.testgensentencepred = True
        self.interface.retrieveaac = False
        self.interface.pathlog = os.path.join(current_dir, "/path/to/logs")
        self.interface.loglevel = logging.INFO

        result = self.interface.createPredictorConfig(config_file_name)

        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result.get("DEFAULT", "static_resources_path"), "/path/to/static")
            self.assertEqual(result.get("DEFAULT", "personalized_resources_path"), "/path/to/personalized")
            self.assertEqual(result.get("DEFAULT", "suggestions"), "5")
            self.assertEqual(result.get("DEFAULT", "test_generalSentencePrediction"), "True")
            self.assertEqual(result.get("DEFAULT", "retrieveAAC"), "False")
            self.assertEqual(result.get("DEFAULT", "log_location"), "/path/to/logs")
            self.assertEqual(result.get("DEFAULT", "log_level"), "INFO")

    def test_handle_learn(self):
        conv_assist = MagicMock()
        messageReceived = MagicMock()
        mode = "WORDS"

        self.interface.handle_learn(conv_assist, messageReceived, mode)

        conv_assist.learn_db.assert_called_once_with(messageReceived.Data)

    def test_handle_parameter_change(self):
        messageReceived = MagicMock()
        messageReceived.Data = '{"Parameter": "SUGGESTIONS", "Value": 10}'

        result = self.interface.handle_parameter_change(messageReceived)

        self.assertTrue(result)
        self.assertEqual(self.interface.suggestions, 10)

    def test_initialize_or_configure_convassists(self):
        convassists = [self.interface.conv_normal, self.interface.conv_shorthand, self.interface.conv_sentence, self.interface.conv_canned_phrases]

        with patch.object(self.interface, "createPredictorConfig") as mock_createPredictorConfig, patch.object(self.interface.logger, "info") as mock_logger_info:
            mock_createPredictorConfig.return_value = MagicMock()
            self.interface.initialize_or_configure_convassists()

            for convassist in convassists:
                if convassist.initialized:
                    convassist.update_params.assert_called_once_with("True", "False")
                    convassist.read_updated_toxicWords.assert_called_once()
                    mock_logger_info.assert_called_with(f"convassist {convassist.id} updated.")
                else:
                    convassist.initialize.assert_called_once()
                    if convassist.id == "SENTENCE":
                        convassist.read_updated_toxicWords.assert_called_once()
                    if convassist.id == "CANNED":
                        convassist.cannedPhrase_recreateDB.assert_called_once()
                    mock_logger_info.assert_called_with(f"convassist {convassist.id} initialized and configured.")

    def test_next_word_prediction(self):
        PredictionResponse = MagicMock()
        messageReceived = MagicMock()
        messageReceived.PredictionType = "NORMAL"
        messageReceived.Data = "test"

        self.interface.conv_normal.initialized = True
        self.interface.conv_normal.context_tracker.callback = MagicMock()
        self.interface.conv_normal.predict = MagicMock(return_value=([], [], [], []))

        self.interface.next_word_prediction(PredictionResponse, messageReceived)

        self.interface.conv_normal.context_tracker.callback.update.assert_called_once_with("test")
        self.interface.conv_normal.context_tracker.prefix.assert_called_once()
        self.interface.conv_normal.context_tracker.past_stream.assert_called_once()
        self.interface.conv_normal.predict.assert_called_once()

    def test_next_sentence_prediction(self):
        PredictionResponse = MagicMock()
        messageReceived = MagicMock()
        messageReceived.Data = "test"

        self.interface.conv_sentence.initialized = True
        self.interface.conv_sentence.check_model = MagicMock(return_value=1)
        self.interface.conv_sentence.context_tracker.callback = MagicMock()
        self.interface.conv_sentence.predict = MagicMock(return_value=([], [], [], []))

        self.interface.next_sentence_prediction(PredictionResponse, messageReceived)

        self.interface.conv_sentence.context_tracker.callback.update.assert_called_once_with("test")
        self.interface.conv_sentence.context_tracker.prefix.assert_called_once()
        self.interface.conv_sentence.context_tracker.past_stream.assert_called_once()
        self.interface.conv_sentence.predict.assert_called_once()

    def test_handle_incoming_messages(self):
        Pipehandle = MagicMock()
        Pipehandle.get_incoming_message = MagicMock(return_value='{"MessageType": "SETPARAM", "Data": "{\"Parameter\": \"SUGGESTIONS\", \"Value\": 10}"}')
        Pipehandle.send_message = MagicMock()
        PredictionResponse = MagicMock()
        ConvAssistMessage = MagicMock()
        ConvAssistSetParam = MagicMock()
        ConvAssistMessage.jsonDeserialize = MagicMock(return_value=ConvAssistSetParam)
        ConvAssistSetParam.Parameter = "SUGGESTIONS"
        ConvAssistSetParam.Value = 10

        with patch.object(self.interface, "handle_parameter_change") as mock_handle_parameter_change, patch.object(self.interface, "initialize_or_configure_convassists") as mock_initialize_or_configure_convassists:
            mock_handle_parameter_change.return_value = True
            self.interface.handle_incoming_messages(Pipehandle)

            Pipehandle.get_incoming_message.assert_called_once()
            ConvAssistMessage.jsonDeserialize.assert_called_once_with('{"MessageType": "SETPARAM", "Data": "{\"Parameter\": \"SUGGESTIONS\", \"Value\": 10}"}')
            mock_handle_parameter_change.assert_called_once_with(ConvAssistSetParam)
            mock_initialize_or_configure_convassists.assert_called_once()
            Pipehandle.send_message.assert_called_once()
