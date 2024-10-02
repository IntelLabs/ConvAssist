# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from configparser import ConfigParser
from typing import Any

from .context_tracker import ContextTracker
from .predictor import (
    CannedPhrasesPredictor,
    SentenceCompletionPredictor,
    SpellCorrectPredictor,
)
from .predictor.smoothed_ngram_predictor import (
    CannedWordPredictor,
    GeneralWordPredictor,
    SmoothedNgramPredictor,
)

predictors = {
    "ShortHandPredictor": SmoothedNgramPredictor,
    "SmoothedNgramPredictor": SmoothedNgramPredictor,
    "CannedWordPredictor": CannedWordPredictor,
    "GeneralWordPredictor": GeneralWordPredictor,
    "SpellCorrectPredictor": SpellCorrectPredictor,
    "SentenceCompletionPredictor": SentenceCompletionPredictor,
    "CannedPhrasesPredictor": CannedPhrasesPredictor,
}


class PredictorRegistry(list):
    """
    Manages instantiation and iteration through predictors and aids in
    generating predictions and learning.

    PredictorRegitry class holds the active predictors and provides the
    interface required to obtain an iterator to the predictors.

    The standard use case is: Predictor obtains an iterator from
    PredictorRegistry and invokes the predict() or learn() method on each
    Predictor pointed to by the iterator.
    """

    def __init__(self):
        super().__init__()

    def set_predictors(
        self,
        config: ConfigParser,
        context_tracker: ContextTracker,
        logger: logging.Logger,
        predictors: list[str] | None = None,
    ):
        self[:] = []

        if not predictors:
            predictors = config.get("PredictorRegistry", "predictors", fallback="").split()

        for predictor in predictors:
            self._add_predictor(predictor, config, context_tracker, logger)

    def _add_predictor(
        self,
        predictor_name,
        config: ConfigParser,
        context_tracker: ContextTracker,
        logger: logging.Logger,
    ):
        predictor: Any = None

        predictor_class = config.get(predictor_name, "predictor_class")

        if predictor_class in predictors:
            try:
                if predictor_class in predictors:
                    predictor = predictors[predictor_class](
                        config, context_tracker, predictor_name, logger
                    )
                else:
                    logger.error(
                        f"Predictor class {predictor_class} is not found in the predictor_mapping dictionary."
                    )

            except TypeError as e:
                logger.error(f"Error instantiating predictor {predictor_name}: {e}")
                return
        if predictor:
            self.append(predictor)

    def model_status(self) -> bool:
        model_status = False

        sentPredictor = self.get_predictor("SentenceCompletionPredictor")
        if type(sentPredictor) is SentenceCompletionPredictor:
            model_status = model_status and sentPredictor.model_loaded

        cannedPredictor = self.get_predictor("CannedPhrasesPredictor")
        if type(cannedPredictor) is CannedPhrasesPredictor:
            model_status = model_status and cannedPredictor.model_loaded

        return model_status

    def get_predictor(self, predictor_name):
        for predictor in self:
            if predictor.predictor_name == predictor_name:
                return predictor
        return None

    def list_predictors(self):
        result = []
        for predictor in predictors:
            if self.get_predictor(predictor):
                result.append(f"{predictor}")

        return result
