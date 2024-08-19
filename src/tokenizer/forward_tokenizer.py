# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from src.tokenizer import Tokenizer
from src.utilities.character import blankspaces, separators

class ForwardTokenizer(Tokenizer):
    def __init__(
        self,
        text,
        blankspaces=blankspaces,
        separators=separators,
    ):
        super().__init__(text, blankspaces, separators)

        self.offend = self.count_characters() - 1
        self.reset_stream()

    def count_tokens(self):
        count = 0
        while self.has_more_tokens():
            count += 1
            self.next_token()

        self.reset_stream()

        return count

    def count_characters(self):
        """
        Counts the number of unicode characters in the IO stream.

        """
        return len(self.text)

    def __next__(self):
        if self.has_more_tokens():
            token = self.next_token()
            if token != "":
                return token
        raise StopIteration

    def __iter__(self):
        return self

    def has_more_tokens(self):
        return self.offset < self.offend

    def next_token(self):
        current: str = self.text[self.offset]
        self.offset += 1
        token = ""

        if self.offset <= self.offend:
            while (
                self.is_blankspace(current) or self.is_separator(current)
            ) and self.offset < self.offend:
                current = self.text[self.offset]
                self.offset += 1

            while (
                not self.is_blankspace(current)
                and not self.is_separator(current)
                and self.offset <= self.offend
            ):

                if self.lowercase=="True":
                    current = current.lower()

                token += current

                current = self.text[self.offset]
                self.offset += 1

                if self.offset > self.offend:
                    current = self.text[-1]
                    if not self.is_blankspace(current) and not self.is_separator(
                        current
                    ):
                        token += current

        return token

    def progress(self):
        return float(self.offset) / self.offend

    def reset_stream(self):
        self.offset = 0
