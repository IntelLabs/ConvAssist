# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import abc

from ConvAssist.utilities.ngram_map import NgramMap
from ConvAssist.utilities.character import blankspaces, separators
from abc import ABCMeta, abstractmethod

class Tokenizer(metaclass=ABCMeta):
    """
    Base class for all tokenizers.
    
    Methods
    is_blankspace(char)

    is_separator(char)

    count_characters()
        Abstract method to count the number of characters in the text.

    reset_stream()
        Abstract method to reset the tokenizer stream.

    count_tokens()
        Abstract method to count the number of tokens in the text.

    has_more_tokens()
        Abstract method to check if there are more tokens in the text.

    next_token()
        Abstract method to get the next token in the text.

    progress()
        Abstract method to track the progress of tokenization.
    """

    def __init__(
        self,
        text,
        blankspaces=blankspaces,
        separators=separators,
    ):
        """
        Constructor of the Tokenizer base class.

        Parameters
        ----------
        text : str
            The text to tokenize.

        blankspaces : str
            The characters that represent empty spaces.

        separators : str
            The characters that separate token units (e.g. word boundaries).

        """
        self.separators = separators
        self.blankspaces = blankspaces
        if(text.startswith(tuple(blankspaces))):
            text = text.strip()
        self.text = text
        self.lowercase = False

        self.offbeg: int = 0
        self.offset: int = 0
        self.offend: int = 0

    def is_blankspace(self, char):
        """
        Test if a character is a blankspace.

        Parameters
        ----------
        char : str
            The character to test.

        Returns
        -------
        ret : bool
            True if character is a blankspace, False otherwise.

        """
        if len(char) > 1:
            raise TypeError("Expected a char.")
        if char in self.blankspaces:
            return True
        return False

    def is_separator(self, char):
        """
        Test if a character is a separator.

        Parameters
        ----------
        char : str
            The character to test.

        Returns
        -------
        ret : bool
            True if character is a separator, False otherwise.

        """
        if len(char) > 1:
            raise TypeError("Expected a char.")
        if char in self.separators:
            return True
        return False

    @abc.abstractmethod
    def count_characters(self) -> int:
        raise NotImplementedError("Method must be implemented")

    @abc.abstractmethod
    def reset_stream(self):
        raise NotImplementedError("Method must be implemented")

    @abc.abstractmethod
    def count_tokens(self) -> int:
        raise NotImplementedError("Method must be implemented")

    @abc.abstractmethod
    def has_more_tokens(self) -> bool:
        raise NotImplementedError("Method must be implemented")

    @abc.abstractmethod
    def next_token(self) -> str:
        raise NotImplementedError("Method must be implemented")

    @abc.abstractmethod
    def progress(self) -> float:
        raise NotImplementedError("Method must be implemented")
