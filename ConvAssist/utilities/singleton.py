# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]
    
class PredictorSingleton(Singleton):
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        else:
            cls._instances[cls].context_tracker = args[1]
        return cls._instances[cls]
