# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from unittest.mock import MagicMock, patch

from ..acatconvassist import (
    ACATConvAssistInterface,
)


class testStaticMethods(unittest.TestCase):
    def setUp(self):
        app_quit_event = MagicMock()
        self.interface = ACATConvAssistInterface(app_quit_event)

    def test_sort_List(self):
        prediction_list = [("a", 0.5), ("b", 0.3), ("c", 0.7)]
        amount_predictions = 2
        expected_result = [("c", 0.7), ("a", 0.5)]
        result = self.interface.sort_List(prediction_list, amount_predictions)
        self.assertEqual(result, expected_result)

    # def test_ConnectToACAT(self):
    #     with patch.object(self.interface.logger, "info") as mock_logger_info, patch.object(self.interface.logger, "error") as mock_logger_error, patch.object(self.interface, "ConnectToNamedPipe") as mock_ConnectToNamedPipe:
    #         mock_ConnectToNamedPipe.return_value = (True, MagicMock())
    #         success, handle = self.interface.ConnectToACAT()

    #         mock_logger_info.assert_called_once_with("Connected to ACAT server.")
    #         self.assertTrue(success)
    #         self.assertIsInstance(handle, MagicMock)

    #         mock_ConnectToNamedPipe.return_value = (False, None)
    #         success, handle = self.interface.ConnectToACAT()

    #         mock_logger_info.assert_called_with("Failed to connect to ACAT server.")
    #         self.assertFalse(success)
    #         self.assertIsNone(handle)

    #         mock_ConnectToNamedPipe.side_effect = Exception("Error")
    #         success, handle = self.interface.ConnectToACAT()

    #         mock_logger_error.assert_called_once()
    #         self.assertFalse(success)
    #         self.assertIsNone(handle)


if __name__ == "__main__":
    unittest.main()
