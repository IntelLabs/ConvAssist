# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Combiner classes to merge results from several predictors.
"""
import abc
from typing import Set

from convassist.predictor.utilities.models.predictions import Predictions
from convassist.predictor.utilities.models.suggestions import MAX_PROBABILITY


class Combiner(metaclass=abc.ABCMeta):
    """
    Base class for all combiners
    """

    def filter(self, predictions: Predictions) -> Predictions:
        seen_tokens: Set[str] = set()
        result = Predictions(predictions.name)
        for i, suggestion in enumerate(predictions):
            token = suggestion.word
            if token not in seen_tokens:
                for j in range(i + 1, len(predictions)):
                    if token == predictions[j].word:
                        # TODO: interpolate here?
                        suggestion.probability += predictions[j].probability
                        if suggestion.probability > MAX_PROBABILITY:
                            suggestion.probability = MAX_PROBABILITY
                seen_tokens.add(token)
                result.add_suggestion(suggestion)
        return result

    @abc.abstractmethod
    def combine(self):
        raise NotImplementedError("Method must be implemented")  # pragma: no cover
