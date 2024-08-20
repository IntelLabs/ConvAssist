# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ConvAssist.tokenizer import Tokenizer
from ConvAssist.utilities.character import blankspaces, separators

class ReverseTokenizer(Tokenizer):
    def __init__(
        self,
        text,
        blankspaces=blankspaces,
        separators=separators,
    ):
        super().__init__(text, blankspaces, separators)
        self.offend = self.count_characters() - 1
        self.offset = self.offend

    def count_tokens(self):
        curroff = self.offset
        self.offset = self.offend
        count = 0
        while self.has_more_tokens():
            self.next_token()
            count += 1
        self.offset = curroff
        return count

    def count_characters(self):
        """
        Counts the number of unicode characters in the IO stream.

        """
        return len(self.text)

    def has_more_tokens(self):
        if self.offbeg <= self.offset:
            return True
        else:
            return False

    def next_token(self):
        token = ""

        while (self.offbeg <= self.offset) and len(token) == 0:
            current: str = self.text[self.offset]

            if (self.offset == self.offend) and (
                self.is_separator(current) or self.is_blankspace(current)
            ):
                self.offset -= 1
                return token

            while (
                self.is_blankspace(current) or self.is_separator(current)
            ) and self.offbeg < self.offset:
                self.offset -= 1
                if self.offbeg <= self.offset:
                    current = self.text[self.offset]

            while (
                not self.is_blankspace(current)
                and not self.is_separator(current)
                and self.offbeg <= self.offset
            ):
                if self.lowercase=="True":
                    current = current.lower()
                token = current + token
                self.offset -= 1
                if self.offbeg <= self.offset:
                    current = self.text[self.offset]

        return token

    def progress(self):
        return float(self.offend - self.offset) / (self.offend - self.offbeg)

    def reset_stream(self):
        self.offset = self.offend
