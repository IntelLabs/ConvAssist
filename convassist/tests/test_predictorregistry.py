# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import logging
import unittest
from unittest.mock import mock_open, patch
from parameterized import parameterized

from convassist.context_tracker import ContextTracker
from convassist.predictor_registry import PredictorRegistry

# from unittest.mock import mock_open, patch


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

    @parameterized.expand(
        [
            ("CannedWordPredictor", "CannedWordPredictor"),
            ("DefaultSmoothedNgramPredictor", "GeneralWordPredictor"),
            ("SpellCorrectPredictor", "SpellCorrectPredictor"),
        ]
    )
    def test_get_predictor_class(self, predictor_name, expected_predictor_class):
        # self.predictor_registry.set_predictors(
        #     self.config, self.context_tracker, logging.getLogger()
        # )
        predictor_class = self.predictor_registry.get_predictor_class(predictor_name, self.config)
        self.assertEqual(predictor_class, expected_predictor_class)


if __name__ == "__main__":
    unittest.main()
