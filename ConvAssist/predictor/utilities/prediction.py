# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from .suggestion import Suggestion


class UnknownCombinerException(Exception):
    pass


class Prediction(list):
    """
    Class for predictions from predictors.

    """

    def add_suggestion(self, suggestion: Suggestion):
        self.append(suggestion)
        self.sort(key=lambda x: x.probability, reverse=True)
