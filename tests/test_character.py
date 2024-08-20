"""
 Copyright (C) <year(s)> Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
import unittest

import convassist.utilities.character as character


class TestCharacter(unittest.TestCase):
    def test_first_word_character(self):
        assert character.first_word_character("8238$§(a)jaj2u2388!") == 7
        assert character.first_word_character("123üäö34ashdh") == 3
        assert character.first_word_character("123&(/==") == -1

    def test_last_word_character(self):
        assert character.last_word_character("8238$§(a)jaj2u2388!") == 13
        assert character.last_word_character("123üäö34ashdh") == 12
        assert character.last_word_character("123&(/==") == -1

    def test_is_word_character(self):
        assert character.is_word_character("ä") == True
        assert character.is_word_character("1") == False
        assert character.is_word_character(".") == False
