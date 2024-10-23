# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import unittest
from unittest.mock import MagicMock

from ..acatconvassist import ACATConvAssistInterface


class TestACATConvAssistInterface(unittest.TestCase):
    def test_init(self):
        app_quit_event = MagicMock()
        interface = ACATConvAssistInterface(app_quit_event)

        self.assertEqual(interface.app_quit_event, app_quit_event)
        self.assertEqual(interface.retries, 5)
        self.assertEqual(interface.pipeName, "ACATConvAssistPipe")
        self.assertIsNotNone(interface.logger)
        self.assertFalse(interface.clientConnected)
        self.assertEqual(interface.path, "")
        self.assertEqual(interface.suggestions, 10)
        self.assertFalse(interface.testgensentencepred)
        self.assertFalse(interface.retrieveaac)
        self.assertEqual(interface.pathstatic, "")
        self.assertEqual(interface.pathpersonalized, "")
        self.assertFalse(interface.enablelogs)
        self.assertEqual(interface.loglevel, logging.DEBUG)
        self.assertEqual(interface._pathlog, "")
        self.assertIsNotNone(interface.word_config)
        self.assertIsNotNone(interface.sh_config)
        self.assertIsNotNone(interface.sent_config)
        self.assertIsNotNone(interface.canned_config)
        self.assertFalse(interface.sent_config_change)
        self.assertTrue(interface.enable_logs)
        self.assertIsNotNone(interface.conv_normal)
        self.assertIsNotNone(interface.conv_shorthand)
        self.assertIsNotNone(interface.conv_sentence)
        self.assertIsNotNone(interface.conv_canned_phrases)


if __name__ == "__main__":
    unittest.main()
