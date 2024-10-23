# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import configparser
import logging
import os
from pathlib import Path

import pyttsx3

from convassist.context_tracker import ContextTracker
from convassist.ConvAssist import ConvAssist

SCRIPT_DIR = str(Path(__file__).resolve().parent)

conv_assist_modes = {
    "default": {
        "description": "Initial configuration based on ini file.",
        "predictors": [],
    },
    "all": {
        "description": "Predicts the next word and sentence using all predictors.",
        "predictors": [
            "SentenceCompletionPredictor",
            "GeneralWordPredictor",
            "SpellCorrectPredictor",
            "CannedPhrasesPredictor",
            "CannedWordPredictor",
            "ShortHandPredictor",
        ],
    },
    "sentence": {
        "description": "Predicts the next word and sentence using sentence and word predictors.",
        "predictors": [
            "SentenceCompletionPredictor",
            "GeneralWordPredictor",
            "SpellCorrectPredictor",
        ],
    },
    "canned": {
        "description": "Predicts the next word and sentence using canned responses.",
        "predictors": [
            "CannedPhrasesPredictor",
            "CannedWordPredictor",
        ],
    },
    "shorthand": {
        "description": "Predicts the next word using shorthand.",
        "predictors": [
            "ShortHandPredictor",
        ],
    },
}


class CustomContextTracker(ContextTracker):
    def __repr__(self):
        return self.context.replace(" ", ".")


class ConvAssistMode:
    def __init__(self, convassist: ConvAssist):
        self.convassist = convassist
        self.mode = conv_assist_modes["default"]

    def list_modes(self):
        for mode in conv_assist_modes:
            print(
                f"{mode}: {conv_assist_modes[mode]['description']} {'(active)' if self.mode == conv_assist_modes[mode] else ''}"
            )

    def set_mode(self, mode: str):
        if mode in conv_assist_modes:
            self.convassist.set_predictors(conv_assist_modes[mode]["predictors"])
            self.mode = conv_assist_modes[mode]
        else:
            raise ValueError(f"Invalid mode: {mode}")


class ContinuousPredict:
    def __init__(self):
        self.ct = CustomContextTracker()

        self.word_predictions = []
        self.sentence_predictions = []

        # config file for continuous predictions
        config_file = os.path.join(SCRIPT_DIR, "resources/continuous_prediction.ini")
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

        success_count = config.read(config_file)

        # Customize file paths
        config["Common"]["home_dir"] = SCRIPT_DIR

        if not success_count:
            print("Config file not found. Exiting.")
            exit(1)

        # Create an instance of ConvAssist
        self.ContinuousPredictor = ConvAssist(
            "CONT_PREDICT", config=config, log_level=logging.ERROR
        )

        # Create an instance of ConvAssistMode
        self.convAssistMode = ConvAssistMode(self.ContinuousPredictor)
        self.convAssistMode.set_mode("sentence")

        self.command_dict = {
            "help": self.show_help,
            "word": self.handle_word,
            "sentence": self.handle_sentence,
            "loglevel": self.set_log_level,
            "context": self.show_context,
            "speak": self.speak_context,
            "learn": self.learn_phrase,
            "mode": self.set_mode,
            "quit": self.quit_program,
        }

    def print_table(self, data):
        headers = ["ID", "Word/Sentence", "Probability"]
        col_widths = [4, 70, 12]
        if headers:
            header = [f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]
            print(" | ".join(header))
            print("-" * (sum(col_widths) + 3 * (len(header) - 1)))

        for index, row in enumerate(data, start=0):
            # format the probability to 2 decimal places as a string
            formatted_row = (f"{index}", row[0], f"{row[1]:.2f}")
            formatted_row = [
                f"{formatted_row[i]:<{col_widths[i]}}" for i in range(len(formatted_row))
            ]
            print(" | ".join(formatted_row))

        print("\n")

    def show_context(self, _):
        print(f"Current context: {self.ct}")

    def show_help(self, _):
        print(
            "Commands: \n"
            ":context - Display the current context. \n"
            ":speak - Speak and learn the current context. \n"
            ":learn <text> - Learn a sentence, word, or phrase. \n"
            ":word <id> - Add a word to the context. (leave blank to print current list) \n"
            ":sentence <id> - Add a sentence to the context. (leave blank to print current list) \n"
            ":loglevel <level> - Set the log level. \n"
            "\t'level' can be one of the following: \n"
            "\tDEBUG, INFO, WARNING, ERROR, CRITICAL \n"
            ":mode <mode> - Set the mode. (leave blank to list available modes).\n"
            ":quit - Quit. \n"
            ":help - Display this help message."
        )

    def handle_word(self, command):
        if len(command) == 1:
            print("Word Predictions")
            self.print_table(self.word_predictions)

        else:
            self.update_context(int(command[1]), self.word_predictions)

    def handle_sentence(self, command):
        if len(command) == 1:
            print("Sentence Predictions")
            self.print_table(self.sentence_predictions)

        else:
            self.update_context(int(command[1]), self.sentence_predictions)

    def update_context(self, index: int, predictions: list):

        last_token = self.ct.get_last_token()
        if last_token == "":  # nosec b105
            self.ct.context += predictions[index][0] + " "
        else:
            last_token_index = self.ct.context.rfind(last_token)
            self.ct.context = self.ct.context[:last_token_index] + predictions[index][0] + " "

        print(f"New context: {self.ct}")
        self.word_predictions, self.sentence_predictions = self.predict()

    def set_log_level(self, command):
        if len(command) == 2:
            level = command[1].upper()
            if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                self.ContinuousPredictor.set_log_level(level)
                print(f"Log level set to: {level}")
            else:
                print(f"Invalid log level: {level}")

    def speak_context(self, _):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1)
        engine.say(self.ct)
        engine.runAndWait()

        self.ContinuousPredictor.learn_text(self.ct.context)

    def learn_phrase(self, command):
        if len(command) > 1:
            phrase = " ".join(command[1:])
            print(f"Learning: {phrase}")
            self.ContinuousPredictor.learn_text(phrase)
        else:
            print("Invalid learn command. Please try again.")

    def set_mode(self, command):
        if len(command) == 2 and command[1]:
            try:
                self.convAssistMode.set_mode(command[1])
                print(f"Mode set to: {command[1]}")
                print("\nAvailable Predictors:", self.ContinuousPredictor.list_predictors())
            except ValueError as e:
                print(e)
        else:
            print("Available modes:")
            self.convAssistMode.list_modes()
            print("\nAvailable Predictors:", self.ContinuousPredictor.list_predictors())

    def predict(self) -> tuple[dict, dict]:
        print("GOING INTO PREDICTION MODE")

        self.ContinuousPredictor.context_tracker.context = self.ct.context
        prefix = self.ContinuousPredictor.context_tracker.token(0)
        context = self.ContinuousPredictor.context_tracker.context
        print(f"PREFIX = {prefix} CONTEXT = {context}")

        (
            _,
            word_predictions,
            _,
            sentence_predictions,
        ) = self.ContinuousPredictor.predict()

        print("Word Predictions")
        self.print_table(word_predictions)
        print("Sentence Predictions")
        self.print_table(sentence_predictions)

        return word_predictions, sentence_predictions

    def quit_program(self, _):
        print("Be seeing you... 👌")
        exit(0)

    def main_loop(self):
        while True:
            buffer = input("Enter text to start predictions.\n(':help' for more commands.)\n")
            userinput = buffer.split(":")
            if len(userinput) > 1 and userinput[0] == "":
                command = userinput[1].split(" ")

                if command and command[0] and command[0] in self.command_dict:
                    self.command_dict[command[0]](command)
                else:
                    print("Unknown command.")
                    continue

            else:
                self.ct.context = buffer
                self.word_predictions, self.sentence_predictions = self.predict()
                print(f"New context: {self.ct}")


def main():
    cp = ContinuousPredict()
    cp.main_loop()


if __name__ == "__main__":
    main()
