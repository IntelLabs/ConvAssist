# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

from convassist.combiner.meritocrity_combiner import MeritocracyCombiner
from convassist.predictor.spell_correct_predictor import SpellCorrectPredictor
from convassist.predictor_activator import PredictorActivator
from convassist.predictor_registry import PredictorRegistry

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later


class TestPredictorActivator(unittest.TestCase):
    def setUp(self):
        self.config = ConfigParser()
        self.config.add_section("Selector")
        self.config.set("Selector", "suggestions", "10")

        self.registry = MagicMock(spec=PredictorRegistry)
        self.context_tracker = MagicMock()
        self.logger = MagicMock()

        self.activator = PredictorActivator(
            self.config, self.registry, self.context_tracker, self.logger
        )

    @patch.object(MeritocracyCombiner, "combine", return_value=([], []))
    def test_predict_no_predictors(self, mock_combine):
        self.registry.__iter__.return_value = []
        result = self.activator.predict()

        self.assertEqual(result, ([], [], [], []))
        self.logger.warning.assert_called_with("No predictors registered.")

    def test_predict_with_predictors(self):

        # mock at least 1 predictor
        predictor_mock = MagicMock()
        predictor_mock.predict.return_value = (["sentence"], ["word"])
        predictor_mock.predictor_name = "MockPredictor"

        # mock the registry
        self.registry.__len__.return_value = 1
        self.registry.__iter__.return_value = [predictor_mock]

        # mock the combiner
        combiner_mock = MagicMock(spec=MeritocracyCombiner)
        combiner_mock.combine.return_value = ([], [])
        self.activator.combiner = combiner_mock

        result = self.activator.predict()

        self.assertEqual(result, ([], [], [], []))
        self.logger.info.assert_any_call(
            "Predictor MockPredictor - Predicted 1 sentences and 1 words"
        )
        combiner_mock.combine.assert_called()

    def test_predict_with_spell_correct_predictor(self):
        predictor_mock = MagicMock()
        predictor_mock.predict.return_value = ([], [])
        predictor_mock.predictor_name = "MockPredictor"

        spell_predictor_mock = MagicMock(spec=SpellCorrectPredictor)
        spell_predictor_mock.predict.return_value = ([], ["corrected_word"])

        self.registry.__len__.return_value = 1
        self.registry.__iter__.return_value = [predictor_mock]
        self.registry.get_predictor.return_value = spell_predictor_mock

        # mock the combiner
        combiner_mock = MagicMock(spec=MeritocracyCombiner)
        combiner_mock.combine.return_value = ([], [])
        self.activator.combiner = combiner_mock

        result = self.activator.predict()

        self.assertEqual(result, ([], [], [], []))
        combiner_mock.combine.assert_called()

    def test_predict_with_exception(self):
        predictor_mock = MagicMock()
        predictor_mock.predict.side_effect = Exception("Test Exception")
        predictor_mock.predictor_name = "MockPredictor"

        self.registry.__len__.return_value = 1
        self.registry.__iter__.return_value = [predictor_mock]
        self.registry.get_predictor.return_value = None

        # mock the combiner
        combiner_mock = MagicMock(spec=MeritocracyCombiner)
        combiner_mock.combine.return_value = ([], [])
        self.activator.combiner = combiner_mock

        result = self.activator.predict()

        self.assertEqual(result, ([], [], [], []))
        self.logger.critical.assert_called_with("Predictor MockPredictor: Test Exception")


if __name__ == "__main__":
    unittest.main()
