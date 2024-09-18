# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import json
import logging
import os
from pathlib import Path

from ConvAssist.ConvAssist import ConvAssist

SCRIPT_DIR = str(Path(__file__).resolve().parent)

# config file for continuous predictions
config_file = os.path.join(SCRIPT_DIR, "resources/continuous_prediction.ini")
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

success_count = config.read(config_file)

if not success_count:
    print("Config file not found. Exiting.")
    exit(1)

# Customize file paths
config["Common"]["home_dir"] = SCRIPT_DIR

# Create an instance of ConvAssist
ContinuousPreidictor = ConvAssist("CONT_PREDICT", config=config, log_level=logging.DEBUG)

convAssists = [ContinuousPreidictor]


def main():
    while True:
        buffer = input("Enter text ('close' to exit): ")
        if buffer == "close":
            print("Closing CLI.")
            break

        print("GOING INTO PREDICTION MODE")

        for convAssist in convAssists:
            convAssist.context_tracker.context = buffer
            prefix = convAssist.context_tracker.token(0)
            context = convAssist.context_tracker.context
            print("PREFIX = ", prefix, " CONTEXT = ", context)

            (
                word_nextLetterProbs,
                word_predictions,
                sentence_nextLetterProbs,
                sentence_predictions,
            ) = convAssist.predict()

            print("word_nextLetterProbs ----", json.dumps(word_nextLetterProbs))
            print("word_predictions: ----- ", json.dumps(word_predictions))
            print("sentence_nextLetterProbs ---- ", json.dumps(sentence_nextLetterProbs))
            print("sentence_predictions: ----- ", json.dumps(sentence_predictions))
            print("---------------------------------------------------")


if __name__ == "__main__":
    main()
