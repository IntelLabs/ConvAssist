# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import logging
import os
from pathlib import Path

from conv_assist_modes import ConvAssistMode, conv_assist_modes

from convassist.ConvAssist import ConvAssist

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
ContinuousPredictor = ConvAssist("CONT_PREDICT", config=config, log_level=logging.ERROR)

# Create an instance of ConvAssistMode
convAssistMode = ConvAssistMode(ContinuousPredictor)


def print_table(data, header=None):
    col_widths = [30, 12]
    if header:
        header = [f"{header[i]:<{col_widths[i]}}" for i in range(len(header))]
        print(" | ".join(header))
        print("-" * (sum(col_widths) + 3 * (len(header) - 1)))

    for row in data:
        # format the probability to 2 decimal places as a string
        formatted_row = (row[0], f"{row[1]:.2f}")
        formatted_row = [f"{formatted_row[i]:<{col_widths[i]}}" for i in range(len(formatted_row))]
        print(" | ".join(formatted_row))

    print("\n")


def main():
    while True:
        buffer = input("Enter text to start predictions.\n('help:' for more commands.)\n")
        command = buffer.split(":")
        if command[0] == "help":
            print(
                "Commands: \n"
                "learn:<text> - Learn a sentence, word, or phrase. \n"
                "loglevel:<level> - Set the log level. \n"
                "\t'level' can be one of the following: \n"
                "\tDEBUG, INFO, WARNING, ERROR, CRITICAL \n"
                "mode:<mode> - Set the mode. (leave blank to list available modes)\n"
                "exit: - Exit the CLI. \n"
                "help: - Display this help message."
            )
            continue

        elif command[0] == "loglevel":
            if len(command) == 2:
                level = command[1].upper()
                if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    ContinuousPredictor.setLogLevel(level)
                    print(f"Log level set to: {level}\n")
                else:
                    print("Invalid log level. Please try again.")
            else:
                print("Invalid log level command. Please try again.")
            continue

        elif command[0] == "learn":
            if len(command) == 2:
                print("Learning: ", command[1])
                ContinuousPredictor.learn_text(command[1])
            else:
                print("Invalid learn command. Please try again.")
            continue
            continue
        elif command[0] == "mode":
            if len(command) == 2 and command[1]:
                try:
                    convAssistMode.set_mode(command[1])
                    print(f"Mode set to: {command[1]}")
                    print("\nAvailable Predictors:", ContinuousPredictor.list_predictors())
                except ValueError as e:
                    print(e)
            else:
                print("Available modes:")
                convAssistMode.list_modes()
                print("\nAvailable Predictors:", ContinuousPredictor.list_predictors())
            continue
        elif command[0] == "exit":
            print("Exiting CLI.")
            break

        print("GOING INTO PREDICTION MODE")

        ContinuousPredictor.context_tracker.context = buffer
        prefix = ContinuousPredictor.context_tracker.token(0)
        context = ContinuousPredictor.context_tracker.context
        print("PREFIX = ", prefix, " CONTEXT = ", context)

        (
            _,
            word_predictions,
            _,
            sentence_predictions,
        ) = ContinuousPredictor.predict()

        print("Word Predictions")
        print_table(word_predictions, header=["Word", "Probability"])
        print("Sentence Predictions")
        print_table(sentence_predictions, header=["Sentence", "Probability"])


if __name__ == "__main__":
    main()
