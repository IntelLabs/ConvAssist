# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type

class MessageHandlerException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message

class MessageHandler(ABC):

    config: dict[str, object] = {}

    def set_config(self, config: dict[str, object]) -> None:
        self.config = config

    @abstractmethod
    def startMessageHandler(self) -> None:
        pass

    @abstractmethod
    def stopMessageHandler(self) -> None:
        pass

    @staticmethod
    def getMessageHandler(config: dict[str, object]) -> MessageHandler:
        if config["type"] == "win32pipe":
            from .Win32PipeHandler import Win32PipeMessageHandler
            raise NotImplementedError("Win32PipeMessageHandler not implemented")

        elif config["type"] == "WebSocket":
            from .WebSocketHandler import WebSocketHandler
            handler = WebSocketHandler()
            handler.set_config(config)
            return handler

        else:
            raise MessageHandlerException("Invalid message handler configuration")
