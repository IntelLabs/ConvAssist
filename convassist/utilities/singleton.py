# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class OptionalSingleton(type):
    """
    A metaclass that allows the class to be a singleton if the singleton attribute is set to True.
    """

    _instances = {}
    _singleton_enabled = False

    def __call__(cls, *args, **kwargs):
        if cls._singleton_enabled:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
        else:
            return super().__call__(*args, **kwargs)

    @classmethod
    def enable_singleton(cls):
        cls._singleton_enabled = True

    @classmethod
    def disable_singleton(cls):
        cls._singleton_enabled = False
