"""
 Copyright (C) 2023 Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
import unittest

import convAssist.character


class TestCharacter(unittest.TestCase):
    def test_first_word_character(self):
        assert convAssist.character.first_word_character("8238$§(a)jaj2u2388!") == 7
        assert convAssist.character.first_word_character("123üäö34ashdh") == 3
        assert convAssist.character.first_word_character("123&(/==") == -1

    def test_last_word_character(self):
        assert convAssist.character.last_word_character("8238$§(a)jaj2u2388!") == 13
        assert convAssist.character.last_word_character("123üäö34ashdh") == 12
        assert convAssist.character.last_word_character("123&(/==") == -1

    def test_is_word_character(self):
        assert convAssist.character.is_word_character("ä") == True
        assert convAssist.character.is_word_character("1") == False
        assert convAssist.character.is_word_character(".") == False
