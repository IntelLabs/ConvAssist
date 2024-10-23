# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import logging
import unittest
from unittest.mock import mock_open, patch

from ..context_tracker import ContextTracker
from ..predictor_registry import PredictorRegistry


class TestPredictorRegistry(unittest.TestCase):
    def setUp(self):
        self.config = configparser.ConfigParser()
        # "PredictorRegistry", "predictors"
        self.config["PredictorRegistry"] = {
            "predictors": "SpellCorrectPredictor",
        }
        self.config["SpellCorrectPredictor"] = {
            "predictor_class": "SpellCorrectPredictor",
            "static_resources_path": "./static/",
            "personalized_resources_path": "./personalized/",
            "spellingdatabase": "big.txt",
        }
        self.config["ContextTracker"] = {"lowercase_mode": "True"}

        self.predictor_registry = PredictorRegistry()
        self.context_tracker = ContextTracker()

    def test_init(self):
        self.assertEqual(len(self.predictor_registry), 0)

    def test_set_predictors(self):
        self.predictor_registry.set_predictors(
            self.config, self.context_tracker, logging.getLogger()
        )
        self.assertEqual(len(self.predictor_registry), 1)

    def test_get_predictor(self):
        self.predictor_registry.set_predictors(
            self.config, self.context_tracker, logging.getLogger()
        )
        predictor = self.predictor_registry.get_predictor("SpellCorrectPredictor")
        self.assertIsNotNone(predictor)

    def test_model_loaded(self):
        self.predictor_registry.set_predictors(
            self.config, self.context_tracker, logging.getLogger()
        )
        self.assertFalse(self.predictor_registry.model_status())


if __name__ == "__main__":
    unittest.main()
