# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from ..models.suggestions import Suggestion


class Predictions(list):

    def __init__(self, name, **args):
        super().__init__(**args)
        self.name = name

    def __repr__(self):
        return f"{self.name}: {super().__repr__()}"

    def add_suggestion(self, suggestion: Suggestion):
        """Add a suggestion to the list of suggestions and sort the list."""
        self.append(suggestion)
        self.sort(key=lambda x: x.probability, reverse=True)

#TODO: FIXME

        # if isinstance(suggestion, Suggestion):
        #     self.append(suggestion)
        #     self.sort(key=lambda x: x.probability, reverse=True)
        # else:
        #     raise TypeError("Only suggestions can be added to predictions.")
