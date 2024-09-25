# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from ..utilities.singleton import OptionalSingleton, Singleton


class TestSingleton(unittest.TestCase):
    def test_singleton(self):
        class TestClass(metaclass=Singleton):
            pass

        instance1 = TestClass()
        instance2 = TestClass()
        self.assertIs(instance1, instance2)

    def test_optional_singleton_disabled(self):
        class TestClass(metaclass=OptionalSingleton):
            pass

        instance1 = TestClass()
        instance2 = TestClass()
        self.assertIsNot(instance1, instance2)

    def test_optional_singleton_enabled(self):
        class TestClass(metaclass=OptionalSingleton):
            pass

        TestClass.enable_singleton()
        instance1 = TestClass()
        instance2 = TestClass()
        self.assertIs(instance1, instance2)

    def test_optional_singleton_enabled_and_disabled(self):
        class TestClass(metaclass=OptionalSingleton):
            pass

        TestClass.enable_singleton()
        instance1 = TestClass()
        instance2 = TestClass()
        self.assertIs(instance1, instance2)

        TestClass.disable_singleton()
        instance3 = TestClass()
        self.assertIsNot(instance1, instance3)
        self.assertIsNot(instance2, instance3)


if __name__ == "__main__":
    unittest.main()
